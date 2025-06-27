from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from model_loader import get_model
from utils import read_image
import tensorflow as tf
import numpy as np

app = FastAPI()

CLASS_NAMES = {
    "dog": ["Demodicosis", "Dermatitis", "Fungal_infections", "Healthy", "Hypersensitivity", "Ringworm"],
    "cat": ["Flea_Allergy", "Health", "Ringworm", "Scabies"],
    "cow": ["Lumpy Skin", "Normal Skin"]
}

@app.post("/predict/")
async def predict(
    animal_type: str = Form(...),
    file: UploadFile = File(...)
):
    if animal_type not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail="Invalid animal_type. Choose from dog, cat, cow.")

    image_bytes = await file.read()
    img_array = read_image(image_bytes)  # Preprocessed image

    model = get_model(animal_type)  # Lazy loaded
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    label = CLASS_NAMES[animal_type][np.argmax(prediction)]

    return JSONResponse(content={"prediction": label})




