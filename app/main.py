import os
import logging
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import RootModel, field_validator
from contextlib import asynccontextmanager
import mlflow

# load environment variables
load_dotenv()

# Configuration
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR", "models")
LOG_DIR = BASE_DIR / os.getenv("LOG_DIR", "log")
EXPECTED_FEATURE_DIM = 8
USE_MLFLOW = os.getenv("USE_MLFLOW", "False").lower() == "true"
DEFAULT_MODEL_LOCAL = os.getenv("DEFAULT_MODEL_LOCAL", "model_v1.pkl")
DEFAULT_MODEL_MLFLOW = os.getenv("DEFAULT_MODEL_MLFLOW", "housing_price_model") 
DEFAULT_MODEL_MLFLOW_ALIAS = os.getenv("DEFAULT_MODEL_MLFLOW_ALIAS", "prod")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set up logging
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level = logging.INFO,
    filename = LOG_DIR / "prediction.log",
    filemode = "a",
    format = "%(asctime)s - %(levelname)s - %(message)s"
)


# Model loading from local file system
def load_model_from_file(model_filename: str):
    model_path = MODEL_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_filename} not found.")
    return joblib.load(model_path)

# Model loading from MLflow
def load_model_from_mlflow(model_name: str, model_alias: str):
    model_uri = f"models:/{model_name}@{model_alias}"
    return mlflow.sklearn.load_model(model_uri)
    

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


def create_app(override_model=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if override_model:
            model = override_model
            model_name = "Overridden Model"
            logging.info(f"Starting FastAPI with overridden model")
        elif USE_MLFLOW:
            model = load_model_from_mlflow(model_name=DEFAULT_MODEL_MLFLOW, model_alias=DEFAULT_MODEL_MLFLOW_ALIAS)
            model_name = f"{DEFAULT_MODEL_MLFLOW}@{DEFAULT_MODEL_MLFLOW_ALIAS}"
            logging.info(f"Starting FastAPI with MLflow model: {model_name}")

        else:
            model = load_model_from_file(DEFAULT_MODEL_LOCAL)
            model_name = DEFAULT_MODEL_LOCAL
            logging.info(f"Starting FastAPI with local model: {model_name}")
        app.state.model = model
        app.state.model_name = model_name
        yield
        
    # Initialize FastAPI
    app = FastAPI(lifespan=lifespan)


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
        if USE_MLFLOW:
            client = mlflow.tracking.MlflowClient()
            model = client.search_registered_models(filter_string=f"name LIKE '{DEFAULT_MODEL_MLFLOW}'")[0]
            return sorted([m.key for m in model.aliases])
        else:
            return sorted([f.name for f in MODEL_DIR.glob("*.pkl")])

    @app.get("/current_model")
    def current_model():
        return {"model_name": app.state.model_name}

    @app.post("/use_model")
    def switch_model(model_version: str):
        if USE_MLFLOW:
            model_uri = f"models:/{DEFAULT_MODEL_MLFLOW}@{model_version}"
            try:
                app.state.model = load_model_from_mlflow(model_uri)
                app.state.model_name = f"{DEFAULT_MODEL_MLFLOW}@{model_version}"
                return {"message": f"Switched to {model_version} of {DEFAULT_MODEL_MLFLOW}"}
            except mlflow.exceptions.MlflowException as e:
                raise HTTPException(status_code=404, detail=str(e))
        else:
            try:
                app.state.model = load_model_from_file(model_version)
                app.state.model_name = model_version
                return {"message": f"Switched to {model_version}"}
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:create_app", factory=True, reload=True)