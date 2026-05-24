import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = (256,256)

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):
            return torch.nn.Sequential(
                torch.nn.Conv2d(a,b,3,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(b,b,3,padding=1),
                torch.nn.ReLU()
            )
        self.d1 = block(3,64)
        self.d2 = block(64,128)
        self.pool = torch.nn.MaxPool2d(2)
        self.u1 = block(128,64)
        self.final = torch.nn.Conv2d(64,1,1)

    def forward(self,x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x = torch.nn.functional.interpolate(x2, scale_factor=2)
        x = self.u1(x)
        return self.final(x)

model = UNet().to(DEVICE)
model.load_state_dict(torch.load("medical_model.pth", map_location=DEVICE))
model.eval()

img = Image.open("medical_data/train_images/136.png").resize(SIZE)
orig = np.array(img)

x = torch.tensor(orig/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = torch.sigmoid(model(x))
    mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

overlay = orig.copy()
overlay[:,:,0] = np.where(mask==1,255,overlay[:,:,0])

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(orig); plt.title("Original")
plt.subplot(1,3,2); plt.imshow(mask,cmap="gray"); plt.title("Mask")
plt.subplot(1,3,3); plt.imshow(overlay); plt.title("Overlay")
plt.axis("off")
plt.show()
