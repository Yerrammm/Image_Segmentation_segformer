import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 CHANGE THESE TO MATCH YOUR TRAINING
MODEL_PATH = "normal_model.pth"   # or unet_model.pth
NUM_CLASSES = 150                 # must match training
INPUT_SIZE = 256                  # must match training size


# ---------------- MODEL ARCHITECTURE (MUST MATCH TRAINING) ---------------- #

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
        self.pool = nn.MaxPool2d(2)
        self.u1 = block(128,64)
        self.final = nn.Conv2d(64,NUM_CLASSES,1)

    def forward(self,x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x = nn.functional.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.u1(x)
        return self.final(x)


model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# ---------------- SEGMENTATION FUNCTION ---------------- #

def run_normal_segmentation(original_path, segmented_path, masked_path):

    img = Image.open(original_path).convert("RGB")
    orig = np.array(img)
    h, w = orig.shape[:2]

    # resize same as training
    img_resized = cv2.resize(orig, (INPUT_SIZE, INPUT_SIZE))

    x = torch.tensor(img_resized/255.0, dtype=torch.float32)
    x = x.permute(2,0,1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x)
        mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

    # resize mask back
    mask = cv2.resize(mask.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)

    # color map
    np.random.seed(42)
    colors = np.random.randint(0,255,(NUM_CLASSES,3),dtype=np.uint8)
    segmented = colors[mask]

    masked = orig.copy()
    masked[mask==0] = [0,0,0]

    cv2.imwrite(segmented_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
    cv2.imwrite(masked_path, cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))

    return "Normal Segmentation Done"
