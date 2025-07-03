import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import create_app, EXPECTED_FEATURE_DIM
import numpy as np


valid_input = [1.0] * EXPECTED_FEATURE_DIM
invalid_input = [1.0, 2.0]

@pytest.fixture(scope="module")
def mock_model():
    model = MagicMock()
    model.predict.side_effect = lambda x: np.array([4.26] * len(x))
    return model

@pytest.fixture(scope="module")
def test_client(mock_model):
    app = create_app(mock_model)
    app.state.model_name = "mock"
    with TestClient(app) as client:
        yield client
        

def test_predict_validation_error(test_client):
    response = test_client.post("/predict", json=invalid_input)
    assert response.status_code == 422

def test_predict_success(test_client):
    response = test_client.post("/predict", json=valid_input)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == 4.26

def test_predict_batch_success(test_client):
    response = test_client.post("/predict_batch", json=[valid_input, valid_input])
    assert response.status_code == 200
    result = response.json()
    assert result["predictions"] == [4.26, 4.26]
