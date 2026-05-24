import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_DIR = "data/voc_2012_segmentation_data/valid_images"
MASK_DIR = "data/voc_2012_segmentation_data/valid_labels"

NUM_CLASSES = 21
BATCH_SIZE = 2
EPOCHS = 2          # continue more
LR = 1e-5           # 🔥 lower for fine-tuning
IMG_SIZE = 256
MAX_SAMPLES = 1000

processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)


class SegDataset(Dataset):
    def __init__(self):
        image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
        self.files = []

        for img_name in image_files:
            base = os.path.splitext(img_name)[0]
            mask_path = os.path.join(MASK_DIR, base + ".png")
            if os.path.isfile(mask_path):
                self.files.append(base)

        random.shuffle(self.files)
        self.files = self.files[:MAX_SAMPLES]

        print("Matched pairs:", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = Image.open(os.path.join(IMG_DIR, name + ".jpg")).convert("RGB")
        mask = Image.open(os.path.join(MASK_DIR, name + ".png"))

        img = img.resize((IMG_SIZE, IMG_SIZE))
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

        mask = np.array(mask, dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = torch.tensor(mask, dtype=torch.long)

        processed = processor(images=img, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze()

        return pixel_values, mask


def main():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    # 🔥 LOAD PREVIOUS TRAINED MODEL
    if os.path.exists("normal_segformer.pth"):
        model.load_state_dict(torch.load("normal_segformer.pth", map_location=DEVICE))
        print("Loaded previous model — continuing fine-tuning")

    loader = DataLoader(
        SegDataset(),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 🔥 Add label smoothing for sharper boundaries
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.05)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, masks in tqdm(loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(pixel_values=imgs)
            logits = outputs.logits

            # 🔥 keep align_corners False for stable edges
            logits = torch.nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            loss = loss_fn(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"FineTune Epoch {epoch+1}/{EPOCHS} Avg Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "normal_segformer.pth")
    print("Fine-tuning completed")


if __name__ == "__main__":
    main()