import os
import logging
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import RootModel, field_validator



# load environment variables
load_dotenv()

# Configuration
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR", "models")
LOG_DIR = BASE_DIR / os.getenv("LOG_DIR", "log")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "model_v1.pkl")
EXPECTED_FEATURE_DIM = 8

# Set up logging
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level = logging.INFO,
    filename = LOG_DIR / "prediction.log",
    filemode = "a",
    format = "%(asctime)s - %(levelname)s - %(message)s"
)


# Initialize FastAPI
app = FastAPI()

# Model loading helper
def load_model_from_file(model_filename: str):
    model_path = MODEL_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_filename} not found.")
    return joblib.load(model_path)

# Load default model
app.state.model = load_model_from_file(DEFAULT_MODEL)
app.state.model_name = DEFAULT_MODEL


# Pydantic input schemas
class ModelInput(RootModel[List[float]]):
    @field_validator("root")
    @classmethod
    def check_length(cls, v):
        if len(v) != EXPECTED_FEATURE_DIM:
            raise ValueError(f"Expected {EXPECTED_FEATURE_DIM} features, got {len(v)}")
        return v

class BatchModelInput(RootModel[List[List[float]]]):
    @field_validator("root")
    @classmethod
    def check_batch_shape(cls, v):
        for row in v:
            if len(row) != EXPECTED_FEATURE_DIM:
                raise ValueError("Each row must have 8 features.")
        return v


# API Endpoints
@app.post("/predict")
def predict(input: ModelInput):
    try: 
        x = [input.root]    
        prediction = app.state.model.predict(x)
        logging.info(f"[{app.state.model_name}] {x} => {prediction[0]}")

        return {
            "input": input.root,
            "prediction": prediction[0]
        }
    except Exception as e:
        logging.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
def predict_batch(batch_input: BatchModelInput):
    try:
        x_batch = batch_input.root
        predictions = app.state.model.predict(x_batch)
        logging.info(f"[{app.state.model_name}] {x_batch} => {predictions.tolist()}")

        return {
            "predictions": predictions.tolist()
        }

    except Exception as e:
        logging.exception(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def list_models():
    return sorted([f.name for f in MODEL_DIR.glob("*.pkl")])

@app.get("/current_model")
def current_model():
    return {"model_name": app.state.model_name}

@app.post("/use_model")
def switch_model(model_name: str):
    try:
        app.state.model = load_model_from_file(model_name)
        app.state.model_name = model_name
        return {"message": f"Switched to {model_name}"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

