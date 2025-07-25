# ML Deploy Demo

This project demonstrates a minimal, end-to-end machine learning deployment system using scikit-learn, FastAPI, and Uvicorn. The service loads a pre-trained model and exposes HTTP endpoints for prediction.

## 📁 Project Structure

```
ML-end-to-end/
├── app/                # FastAPI application code
│   └── main.py         # API endpoints and model serving logic
├── models/             # Trained model files (model_v1.pkl, model_v2.pkl)
├── notebooks/          # One-off scripts and notebooks like model training
│   ├── 01.train.ipynb
│   └── 02.train_mlflow.ipynb
├── log/                # Log files generated by FastAPI service
│   └── prediction.log
├── tests/              # test functions
│   ├── test_model.py
│   ├── test_api.py
│   └── test_api_mocked.py
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build file
├── docker-compose.yml  # Container service configuration
├── .env                # Environment variable file (not committed)
├── pytest.ini          # Pytest config for Python path
└── README.md           # Project documentation
```

## 🚀 Quick Start

### 0. Define environment variables
Create a `.env` file in the project root:
```bash
MODEL_DIR=models
LOG_DIR=log
DEFAULT_MODEL_LOCAL=model_v1.pkl
```

### 1. Set up virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Train and save model
Refer to `notebooks/01.train.ipynb` and `notebooks/01.train_mlflow.ipynb` 

### 3. Run with Docker Compose
```bash
docker compose up --build
```

Visit Swagger UI at: [http://localhost:8000/docs](http://localhost:8000/docs)


## 📒 Using MLflow

This project supports both local model loading and **MLflow Model Registry**.

To use MLflow:
1. Set the following in `.env`:
   ```env
   USE_MLFLOW=true
   MLFLOW_TRACKING_URI=file:///app/mlruns
   DEFAULT_MODEL_MLFLOW=housing_price_model
   DEFAULT_MODEL_MLFLOW_ALIAS=prod

2. Train and register model (recommended: run inside Docker):
```bash
docker compose exec ml-api-demo bash
python notebooks/02.train_mlflow.py
```

3. Verify in UI:
```bash
mlflow ui --backend-store-uri mlruns
```

4. When running FastAPI, it will automatically pull the registered model based on the alias.
❗ Tip: Make sure to **train and register from inside the container**, so that the artifact paths are valid.


## 🔍 API Endpoints

### `/predict` (POST)
Predict on a single input vector of 8 features (adjust as needed).
```json
[8.3252,41,6.98412698,1.02380952,322,2.55555556,37.88,-122.23]
```
Response:
```json
{
  "input": [8.3252,41,6.98412698,1.02380952,322,2.55555556,37.88,-122.23],
  "prediction": 4.265793
}
```

### `/predict_batch` (POST)
Predict on multiple input vectors.
```json
[
  [8.3252,41,6.98412698,1.02380952,322,2.55555556,37.88,-122.23],
  [7.2574,52,8.28813559,1.07344633,496,2.80225989,37.85, -122.24]
]
```
Response:
```json
{
  "predictions": [4.265793, 3.75166]
}
```

### `/models` (GET)
Returns available model filenames under `models/`.

### `/use_model` (POST)
Switch to a different model file.
```json
{"model_name": "model_v2.pkl"}
```


## 🪵 Logging
All requests and predictions are logged to `log/prediction.log` with timestamps.


## 🧪 Testing
### 1. Install pytest
```bash
pip install pytest
```

### 2. Run tests
```bash
pytest tests/
```

Test cases include:
- Model loading and shape validation
- FastAPI endpoints (/predict, /predict_batch, /use_model)
- Input validation and expected failure responses
- Mock-based tests for CI pipelines


## 🐳 Docker Notes
If you're using Docker directly:
```dockerfile
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```
Note: Factory loading is required when using FastAPI's lifespan startup.


## 🧱 Tech Stack
- Python 3.10+
- scikit-learn
- FastAPI
- Uvicorn
- joblib
- numpy
- python-dotenv
- Docker
- Docker Compose
- Pytest
- GitHub Actions
- MLflow

## 📌 Future Extensions
- .env multi-environment support (.env.dev, .env.prod)
- MLflow remote tracking server
- Streamlit or React frontend
- Gunicorn + Nginx production deployment

## 🛠️ Development Log
### 06/15/2025 Updates
- ✅ Created model training pipeline
- ✅ Set up Fast API + Uvicorn local service, supporting single and batch prediction
- ✅ Added exception handling and logging

### 06/22/2025 Updates
- ✅ Completed containerization with docker and docker-compose
- ✅ Added basic unit tests for model prediction
- ✅ Set up volume mount for model and log files

### 06/29/2025 Updates
- ✅ Added comprehensive API tests using `fastapi.testclient`
- ✅ Updated unit tests for model loading, prediction, and failure modes
- ✅ Configured dynamic `BASE_DIR` + `.env` variable for path management
- ✅ Enabled dynamic model switching & model directory scanning
- ✅ Introduced `Pydantic`-based input validation with `field_validator`
- ✅ Fixed module resolution for pytest via `pytest.ini` and `sys.path` patching
- ✅ All tests passing; preparing for CI/CD and MLflow integration

### 06/30/2025 Updates
- ✅ Refactored to use `create_app()` with lifespan for safe model loading.
- ✅ Added mock-based tests (`test_api_mocked.py`) to support CI without real model files.
- ✅ Integrated GitHub Actions for CI with separate test workflows.
- ✅ Fixed test failures by aligning TestClient with async lifespan.
- ✅ Updated Docker config to support FastAPI factory (--factory flag).

### 07/06/2025 Updates
- ✅ Set up local MLflow tracking server, monitor the training process, record the parameters and metrics, and register models
- ✅ Got familiar with experiments, runs, model register, model tag with MLflow
- ✅ Updated main.py to support model loading from either MLflow registry or local files, use environment variable to 

#### 📌 Reflection & Notes
- ❗ Avoid registering models on the host if the container will be loading them, since artifact_location will be saved as an absolute host path and cannot be resolved inside the container
- ✅ The correct approach is to train and register models inside the container, or use a shared remote backend
- ✅ With FastAPI + MLflow integration, model loading logic should be handled inside a lifespan function to ensure proper startup behavior


## 📚 Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Lifespan Docs](https://fastapi.tiangolo.com/advanced/events/)
- [Docker Compose Reference](https://docs.docker.com/compose/)


###
---

Feel free to fork, adapt, and build on this template!

