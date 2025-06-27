import os
import threading
import tensorflow as tf
from gdown import download  # install with `pip install gdown`

_model_locks = {
    "dog": threading.Lock(),
    "cat": threading.Lock(),
    "cow": threading.Lock()
}
_models = {}

GDRIVE_URLS = {
    "dog": "https://drive.google.com/uc?id=1Mk1C9wP422L1IFqka2ynfIhd80Lyc2O5",
    "cat": "https://drive.google.com/uc?id=1JpYlQIJUJFfkBIYO0i3OelZOvdGAj4Bk",
    "cow": "https://drive.google.com/uc?id=1nTI-XnFoFtAsh5uF2t0RxreNmkJnV4Fg"
}

def get_model(animal_type: str):
    if animal_type in _models:
        return _models[animal_type]

    with _model_locks[animal_type]:
        if animal_type not in _models:
            model_path = f"{animal_type}.keras"
            if not os.path.exists(model_path):
                print(f"⬇️ Downloading {animal_type}.keras from GDrive...")
                gdrive_url = GDRIVE_URLS[animal_type]
                download(gdrive_url, model_path, quiet=False)
            
            print(f"🔄 Loading model for: {animal_type}")
            model = tf.keras.models.load_model(model_path)
            _models[animal_type] = model
        return _models[animal_type]
