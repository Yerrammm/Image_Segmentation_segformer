import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

IMG_DIR = "output_voc/images/train"
MASK_DIR = "output_voc/masks/train"

BATCH_SIZE = 4
EPOCHS = 25
NUM_CLASSES = 7
LR = 3e-4

# ---------- DATASET ---------- #

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.names = [f.replace(".npy","") for f in os.listdir(img_dir)]
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]

        img = np.load(f"{self.img_dir}/{name}.npy").astype(np.float32)
        mask = np.array(Image.open(f"{self.mask_dir}/{name}.png"))

        img = img / 255.0

        if np.random.rand() > 0.5:
            img = np.flip(img,1).copy()
            mask = np.flip(mask,1).copy()

        img = torch.tensor(img).permute(2,0,1)
        mask = torch.tensor(mask, dtype=torch.long)

        return img, mask

loader = DataLoader(
    SegDataset(IMG_DIR, MASK_DIR),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ---------- U-NET ---------- #

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(a,b):
            return nn.Sequential(
                nn.Conv2d(a,b,3,padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
                nn.Conv2d(b,b,3,padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True)
            )

        self.d1 = block(3,64)
        self.d2 = block(64,128)
        self.d3 = block(128,256)
        self.pool = nn.MaxPool2d(2)

        self.u1 = block(256,128)
        self.u2 = block(128,64)

        self.final = nn.Conv2d(64, NUM_CLASSES, 1)

    def forward(self,x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))

        x = nn.functional.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.u1(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.u2(x)

        return self.final(x)

# ---------- TRAIN ---------- #

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=255)
opt = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total = 0

    for imgs, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        imgs, masks = imgs.to(device), masks.to(device)

        pred = model(imgs)
        loss = loss_fn(pred, masks)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()

    print(f"Loss: {total:.4f}")

torch.save(model.state_dict(),"unet_model.pth")
print("Model saved!")
