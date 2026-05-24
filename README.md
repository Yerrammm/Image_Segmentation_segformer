# Image Segmentation using UNet and SegFormer

This project performs image segmentation using:
- UNet
- SegFormer

The project supports:
- Medical image segmentation
- Normal object segmentation
- Flask web application for prediction

## Features

- Train UNet models
- Train SegFormer models
- Medical image prediction
- Flask-based web UI
- Upload and visualize segmentation results

## Project Structure

```bash
app.py
train_unet.py
train_medical.py
train_normal_segformer.py
predict_medical.py
templates/
static/
```

## Installation

Clone repository:

```bash
git clone https://github.com/Yerrammm/Image_Segmentation_segformer.git
cd Image_Segmentation_segformer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Flask App

```bash
python app.py
```

## Model Weights

Large model files (.pth) are not included in this repository.

## Technologies Used

- Python
- PyTorch
- Flask
- OpenCV
- SegFormer
- UNet
