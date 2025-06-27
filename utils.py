import gdown
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

def download_from_gdrive(file_id, dest_path):
    """
    Downloads a file from Google Drive using gdown.
    """
    gdown.download(id=file_id, output=dest_path, quiet=False)

def read_image(image_bytes):
    """
    Reads and preprocesses the uploaded image.
    Applies ResNet50 preprocessing.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array
