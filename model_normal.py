import matplotlib
matplotlib.use('Agg')   # 🔥 FIX (VERY IMPORTANT)

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 768

processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)

model.to(DEVICE)
model.eval()


# ===== CONFUSION MATRIX =====
def generate_confusion_matrix(pred_mask, save_path):
    classes = np.unique(pred_mask)

    cm = np.zeros((len(classes), len(classes)))

    for i, c in enumerate(classes):
        cm[i][i] = np.sum(pred_mask == c)

    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Pseudo)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ===== MAIN FUNCTION =====
def run_normal_segmentation(original_path, segmented_path, masked_path):

    img = Image.open(original_path).convert("RGB")
    original = np.array(img)

    resized = img.resize((IMG_SIZE, IMG_SIZE))

    inputs = processor(images=resized, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

        logits = torch.nn.functional.interpolate(
            logits,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )

        probs = torch.softmax(logits, dim=1)
        confidence = torch.mean(torch.max(probs, dim=1)[0]).item()

        pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()

    pred_mask = cv2.resize(
        pred_mask.astype(np.uint8),
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    kernel = np.ones((3, 3), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

    np.random.seed(0)
    color_map = np.random.randint(0, 255, (150, 3))
    colored_mask = color_map[pred_mask].astype(np.uint8)

    overlay = cv2.addWeighted(
        original.astype(np.uint8),
        0.65,
        colored_mask,
        0.35,
        0
    )

    cv2.imwrite(segmented_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    cv2.imwrite(masked_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # CONFUSION MATRIX
    cm_path = segmented_path.replace("seg_", "cm_")
    generate_confusion_matrix(pred_mask, cm_path)

    return round(confidence * 100, 2), cm_path