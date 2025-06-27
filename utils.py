import requests
import numpy as np
from PIL import Image
from io import BytesIO

def download_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def read_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # Change size to match training
    return np.array(img) / 255.0  # Normalize if trained that way
