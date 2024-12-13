import os
import keras
from PIL import Image
import numpy as np


# Directory containing logo images
DATASET_PATH = "D:\\TrainingData\\LogoGeneratorTrainData\\logo_data"


# Load and preprocess images
def load_images(dataset_path, image_size=64):
    images = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = Image.open(os.path.join(dataset_path, filename)).convert("RGB")
            img = img.resize((image_size, image_size))  # Resize to 64x64
            img = keras.utils.image.img_to_array(img) / 127.5 - 1.0  # Normalize to [-1, 1]
            images.append(img)
    return np.array(images)
