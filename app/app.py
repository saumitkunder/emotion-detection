
import streamlit as st
import numpy as np
import tensorflow as tf
import os

# Handle imports for both local testing and Streamlit context
if __name__ == "__main__":
    from utils import preprocess_audio, get_emotion_label
else:
    from app.utils import preprocess_audio, get_emotion_label

MODEL_PATH = "/Users/saumit/Projects/emotion-detection/model/emotion_rnn_model.h5"

@st.cache_resource
def load_model():
    """
    Load the pre-trained RNN model for emotion detection.
    Returns:
        model: The loaded TensorFlow/Keras model.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at {MODEL_PATH}. Please check!")
            return None
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model. Error: {e}")
        return None

# Streamlit UI
st.title("Audio Emotion Detection")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    try:
        st.audio(uploaded_file, format="audio/wav")
        # Preprocess the uploaded audio
        features = preprocess_audio(uploaded_file)
        model = load_model()
        if model is not None:
            predictions = model.predict(features)
            predicted_emotion = get_emotion_label(np.argmax(predictions))
            st.success(f"Predicted Emotion: {predicted_emotion}")
        else:
            st.error("Model failed to load. Please check the logs.")
    except ValueError as ve:
        st.error(f"Error in processing the uploaded file: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


