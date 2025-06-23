# test_predict.py

import requests
import pytest

BASE_URL = "http://127.0.0.1:8000"

@pytest.fixture(scope="module")
def client():
    return requests.Session()

def test_single_predict(client):
    input_data = [8.3252,41,6.98412698,1.02380952,322,2.55555556,37.88,-122.23]
    response = client.post(f"{BASE_URL}/predict", json=input_data)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert isinstance(result["prediction"], float)
    assert result["prediction"] == pytest.approx(4.26579, rel=1e-2) 

def test_batch_predict(client):
    batch_data = [
        [8.3252,41,6.98412698,1.02380952,322,2.55555556,37.88,-122.23],
        [7.2574,52,8.28813559,1.07344633,496,2.80225989,37.85, -122.24]
    ]
    response = client.post(f"{BASE_URL}/predict_batch", json=batch_data)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert isinstance(result["predictions"], list)
    assert all(isinstance(p, float) for p in result["predictions"])
