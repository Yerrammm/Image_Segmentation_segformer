from PIL import Image
import numpy as np

mask = Image.open("data/voc_2012_segmentation_data/valid_labels/2007_001311.png")
mask = np.array(mask)

print("Unique values:", np.unique(mask))