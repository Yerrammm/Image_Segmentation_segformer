import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_DIR = "medical_data/train_images"
MASK_DIR = "medical_data/train_masks"

IMG_SIZE = 256
BATCH_SIZE = 6
EPOCHS = 15
LR = 1e-4
MAX_SAMPLES = 2000


class MedicalDataset(Dataset):
    def __init__(self):
        files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
        random.shuffle(files)
        self.files = files[:MAX_SAMPLES]
        print("Using samples:", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = Image.open(os.path.join(IMG_DIR, name)).convert("RGB")
        mask = Image.open(os.path.join(MASK_DIR, name)).convert("L")

        img = img.resize((IMG_SIZE, IMG_SIZE))
        mask = mask.resize((IMG_SIZE, IMG_SIZE))

        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 127).astype(np.float32)

        img = torch.tensor(img).permute(2, 0, 1)
        mask = torch.tensor(mask).unsqueeze(0)

        return img, mask


def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = double_conv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = double_conv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = double_conv(64, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = double_conv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = double_conv(64, 32)

        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        p1 = self.pool1(c1)

        c2 = self.enc2(p1)
        p2 = self.pool2(c2)

        bridge = self.bridge(p2)

        u1 = self.up1(bridge)
        u1 = torch.cat([u1, c2], dim=1)
        c3 = self.dec1(u1)

        u2 = self.up2(c3)
        u2 = torch.cat([u2, c1], dim=1)
        c4 = self.dec2(u2)

        return self.final(c4)


def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))


def main():
    dataset = MedicalDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, masks in tqdm(loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(imgs)
            loss = 0.5 * bce(outputs, masks) + 0.5 * dice_loss(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} Avg Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "medical_unet_strong.pth")
    print("Strong Medical Model Saved")


if __name__ == "__main__":
    main()