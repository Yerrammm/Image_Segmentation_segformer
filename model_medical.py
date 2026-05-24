import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256


# ================= MODEL =================
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


# ================= LOAD MODEL =================
model = UNet().to(DEVICE)
model.load_state_dict(torch.load("medical_unet_strong.pth", map_location=DEVICE))
model.eval()


# ================= CONFUSION MATRIX =================
def generate_medical_cm(mask, save_path):

    bg = np.sum(mask == 0)
    tumor = np.sum(mask == 1)

    cm = np.array([
        [bg, 0],
        [0, tumor]
    ])

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Reds")
    plt.title("Medical Confusion Matrix (Pseudo)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]),
                     ha="center", va="center", color="black")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ================= MAIN FUNCTION =================
def run_medical_segmentation(original_path, segmented_path, masked_path):

    img = Image.open(original_path).convert("RGB")
    original = np.array(img)

    resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

    x = torch.tensor(resized / 255.0, dtype=torch.float32)
    x = x.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(x))
        prob = pred.squeeze().cpu().numpy()

        mask = (prob > 0.45).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Resize back
    mask = cv2.resize(mask, (original.shape[1], original.shape[0]),
                      interpolation=cv2.INTER_NEAREST)

    # Overlay
    overlay = original.copy()
    overlay[mask == 1] = [255, 0, 0]

    # Save outputs
    cv2.imwrite(segmented_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(masked_path, mask * 255)

    # ===== CONFUSION MATRIX =====
    cm_path = segmented_path.replace("seg_", "cm_")
    generate_medical_cm(mask, cm_path)

    return "Medical Segmentation Done", cm_path