import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
import numpy as np
from PIL import Image
import cv2
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained SegFormer (ADE20K dataset)
processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
).to(DEVICE)

model.eval()


def run_segmentation(original_path, segmented_path, masked_path):

    # Load image
    img = Image.open(original_path).convert("RGB")
    orig = np.array(img)

    # Preprocess
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Resize logits back to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=img.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        probs = torch.softmax(upsampled_logits, dim=1)
        confidence, mask = probs.max(dim=1)

    mask = mask.squeeze().cpu().numpy()
    confidence = confidence.squeeze().cpu().numpy()

    # ---------------- Multi-Class Color Mapping ----------------
    num_classes = mask.max() + 1
    np.random.seed(42)
    colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)

    # Highlight walls clearly (ADE20K wall class id = 2)
    if num_classes > 2:
        colors[2] = [255, 0, 0]   # Red for walls

    segmented = colors[mask]

    # ---------------- Masked Image ----------------
    masked = orig.copy()
    masked[mask == 0] = [0, 0, 0]  # Remove background

    # Save results
    cv2.imwrite(segmented_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
    cv2.imwrite(masked_path, cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))

    accuracy = float(confidence.mean() * 100)
    return round(accuracy, 2)
