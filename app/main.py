from fastapi import FastAPI
from pathlib import Path
import joblib
import numpy as np
from typing import List
import os
import logging
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent.parent

# Step 1: Create FastAPI app
app = FastAPI()

# Step 2: Set up logging to file
# logging.basicConfig(level=logging.INFO)
LOG_DIR = BASE_DIR / "log"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=BASE_DIR / "log" / "service.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Step 3: Load the pre-trained model
# # Resolve the path to model.pkl relative to this file
# MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")
# model = joblib.load(MODEL_PATH)

# Read the model from the environment variable
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "model.pkl"))
model = joblib.load(MODEL_PATH)

# Step 3: Define a prediction endpoint
@app.post("/predict")
def predict(input: List[float]):
    data = np.array([input])
    pred = model.predict(data)
    return {"prediction": pred[0]}

@app.post("/predict")
def predict(input_vector: List[float]):
    try:
        if len(input_vector) != 8:
            raise ValueError("Input vector must have exactly 8 elements.")

        input_array = np.array([input_vector])
        prediction = model.predict(input_array)

        logging.info(f"Single prediction input: {input_vector} | Output: {prediction[0]}")

        return {
            "input": input_vector,
            "prediction": float(prediction[0])
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
def predict_batch(batch_input: List[List[float]]):
    try:
        input_array = np.array(batch_input)
        predictions = model.predict(input_array)

        logging.info(f"Batch prediction input: {batch_input} | Output: {predictions.tolist()}")

        return {
            "predictions": predictions.tolist()
        }

    except Exception as e:
        logging.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
