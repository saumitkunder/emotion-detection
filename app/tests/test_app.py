import pytest
from app.app import load_model

def test_load_model():
    """
    Test the load_model function to ensure the model is loaded correctly.
    """
    model = load_model()
    assert model is not None, "Model failed to load"
