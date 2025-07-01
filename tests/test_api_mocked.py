import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app, EXPECTED_FEATURE_DIM

client = TestClient(app)

valid_input = [1.0] * EXPECTED_FEATURE_DIM
invalid_input = [1.0, 2.0]

def test_predict_validation_error():
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422

@patch("app.main.model_manager.predict", return_value=4.26)
def test_predict_success(mock_predict):
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == 4.26

@patch("app.main.model_manager.predict_batch", return_value=[4.26, 3.75])
def test_predict_batch_success(mock_batch):
    response = client.post("/predict_batch", json=[valid_input, valid_input])
    assert response.status_code == 200
    result = response.json()
    assert result["predictions"] == [4.26, 3.75]

@patch("app.main.model_manager.predict", return_value=4.26579)
def test_single_predict(mock_predict):
    input_data = [8.3252,41,6.98412698,1.02380952,322,2.55555556,37.88,-122.23]
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == pytest.approx(4.26579, rel=1e-2)

@patch("app.main.model_manager.predict_batch", return_value=[4.26579, 3.75166])
def test_batch_predict(mock_batch):
    batch_data = [
        [8.3252,41,6.98412698,1.02380952,322,2.55555556,37.88,-122.23],
        [7.2574,52,8.28813559,1.07344633,496,2.80225989,37.85, -122.24]
    ]
    response = client.post("/predict_batch", json=batch_data)
    result = response.json()
    assert result["predictions"] == pytest.approx([4.26579, 3.75166], rel=1e-2)