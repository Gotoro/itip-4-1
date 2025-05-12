import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI, File, UploadFile
from io import BytesIO

app = FastAPI()

model_path = "laptop_price_pipeline.pkl"
try:
    model = joblib.load(model_path)
    print(f"Модель '{model_path}' успешно загружена.")
except FileNotFoundError:
    print(f"Файл модели '{model_path}' не найден!")
    model = None
except Exception as e:
    print(f"Ошибка при загрузке модели '{model_path}': {e}")
    model = None

@app.get("/")
async def root():
    if model:
        return {"message": "Модель загружена."}
    else:
        return {"message": "ОШИБКА: Модель НЕ загружена."}

@app.post("/predict/")
# без uploadfile не работает fastapi
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Модель не загружена."}

    content = await file.read()
    df = pd.read_csv(BytesIO(content))

    actual_predictions = np.expm1(model.predict(df))

    return {"predictions": actual_predictions.tolist()}
