import pytest
from app.app import load_model
import os

def load_model():
    model_path = "emotion-detection/model/emotion_rnn_model.h5"  # Adjust this path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load the model using your preferred library (e.g., TensorFlow, PyTorch)
    try:
        from tensorflow.keras.models import load_model as tf_load_model
        model = tf_load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

