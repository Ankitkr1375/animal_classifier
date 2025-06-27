import gdown
import numpy as np
from PIL import Image
from io import BytesIO

def download_from_gdrive(file_id, dest_path):
    """
    Downloads a file from Google Drive using gdown.
    """
    gdown.download(id=file_id, output=dest_path, quiet=False)

def read_image(image_bytes):
    """
    Reads and preprocesses the uploaded image.
    Resizes to 224x224 and normalizes pixel values.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # Resize to match model input size
    return np.array(img) / 255.0  # Normalize to [0, 1] range
