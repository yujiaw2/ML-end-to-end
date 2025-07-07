# tests/test_model.py
import pytest
from app.main import load_model_from_file, EXPECTED_FEATURE_DIM, DEFAULT_MODEL_LOCAL, MODEL_DIR

MODEL_PATH = MODEL_DIR / DEFAULT_MODEL_LOCAL

@pytest.fixture(scope="module")
def model():
    return load_model_from_file(DEFAULT_MODEL_LOCAL)

def test_model_file_exists():
    assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"

def test_model_prediction_shape(model):
    input_vector = [[1.0] * EXPECTED_FEATURE_DIM]
    pred = model.predict(input_vector)
    assert len(pred) == 1
    assert isinstance(pred[0], float) or hasattr(pred[0], 'item')

def test_model_raises_on_wrong_dim(model):
    bad_input = [[1.0, 2.0]]  # too short
    with pytest.raises(ValueError):
        model.predict(bad_input)
