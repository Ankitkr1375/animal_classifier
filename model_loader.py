import os
import tensorflow as tf
from utils import download_from_gdrive
import tempfile
models = {}

GDRIVE_FILE_IDS = {
    "dog": "1Mk1C9wP422L1IFqka2ynfIhd80Lyc2O5",
    "cat": "1JpYlQIJUJFfkBIYO0i3OelZOvdGAj4Bk",
    "cow": "1nTI-XnFoFtAsh5uF2t0RxreNmkJnV4Fg"
}

def get_model(animal):
    if animal in models:
        return models[animal]

    tmp_path = os.path.join(tempfile.gettempdir(), f"{animal}.keras")

    if not os.path.exists(tmp_path):
        print(f"Downloading {animal} model...")
        download_from_gdrive(GDRIVE_FILE_IDS[animal], tmp_path)

    model = tf.keras.models.load_model(tmp_path)
    models[animal] = model
    return model
