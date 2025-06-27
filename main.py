from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from model_loader import get_model
from utils import read_image
import tensorflow as tf
import numpy as np
import uvicorn

app = FastAPI()

CLASS_NAMES = {
    "dog": ["Demodicosis", "Dermatitits", "Fungal_Infection", "Healthy", "Hypersenstivity", "Ringworm"],
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
    img_array = read_image(image_bytes)

    model = get_model(animal_type)  # Lazy load model
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    label_index = np.argmax(prediction)
    label = CLASS_NAMES[animal_type][label_index]

    # 🔍 Debug info
    print(f"Prediction raw values: {prediction}")
    print(f"Predicted index: {label_index}, label: {label}")

    return JSONResponse(content={"prediction": label})
