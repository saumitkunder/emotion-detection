import os
import streamlit as st
import tensorflow as tf
import numpy as np
from utils import preprocess_audio, get_emotion_label

# Path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "emotion_rnn_model.h5")

# Load the model with caching
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please check!")
        return None
    try:
        # Load the fixed model
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model. Error: {str(e)}")
        return None
    return model

# Load the model
model = load_model()

# Streamlit app UI
st.title("Audio Emotion Detection App")
st.write("Upload an audio file to detect emotions.")

# File uploader for audio input
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Display the uploaded audio
    st.audio(uploaded_file, format="audio/wav")

    # Preprocess the audio file
    st.write("Processing...")
    try:
        audio_array = preprocess_audio(uploaded_file)
        st.write(f"Preprocessed input shape: {audio_array.shape}")
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Make predictions
    if model:
        try:
            predictions = model.predict(audio_array)
            emotion = get_emotion_label(np.argmax(predictions))
            st.write(f"Predicted Emotion: {emotion}")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    else:
        st.error("Model could not be loaded.")
