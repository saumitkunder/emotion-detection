from app.app import load_model

def test_load_model():
    model = load_model()
    assert model is not None, "Model failed to load"
