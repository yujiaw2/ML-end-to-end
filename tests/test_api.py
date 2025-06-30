import pytest
from fastapi.testclient import TestClient
from app.main import app, EXPECTED_FEATURE_DIM

client = TestClient(app)

# Sample valid input
valid_input = [1.0] * EXPECTED_FEATURE_DIM

# Sample invalid input
invalid_input = [1.0, 2.0]

def test_predict_success():
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_validation_error():
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422

def test_predict_batch_success():
    response = client.post("/predict_batch", json=[valid_input, valid_input])
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 2

def test_model_switch():
    response = client.get("/models")
    assert response.status_code == 200
    available_models = response.json()
    if available_models:
        target = available_models[0]
        switch_response = client.post("/use_model", params={"model_name": target})
        assert switch_response.status_code == 200
        assert f"Switched to {target}" in switch_response.json()["message"]

def test_single_predict():
    input_data = [8.3252,41,6.98412698,1.02380952,322,2.55555556,37.88,-122.23]
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert isinstance(result["prediction"], float)
    assert result["prediction"] == pytest.approx(4.26579, rel=1e-2)

def test_batch_predict():
    batch_data = [
        [8.3252,41,6.98412698,1.02380952,322,2.55555556,37.88,-122.23],
        [7.2574,52,8.28813559,1.07344633,496,2.80225989,37.85, -122.24]
    ]
    response = client.post("/predict_batch", json=batch_data)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) == 2
    assert result["predictions"][0] == pytest.approx(4.26579, rel=1e-2)
    assert result["predictions"][1] == pytest.approx(3.75166, rel=1e-2) 
