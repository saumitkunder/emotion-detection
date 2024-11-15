import io
import numpy as np
import soundfile as sf
from app.utils import preprocess_audio, get_emotion_label

def test_preprocess_audio():
    """
    Test the preprocess_audio function by simulating an audio file and checking output shape.
    """
    # Create a fake audio file
    fake_audio = np.random.random(22050 * 5).astype(np.float32)  # 5 seconds of fake audio
    buffer = io.BytesIO()
    sf.write(buffer, fake_audio, samplerate=22050, format='WAV')
    buffer.seek(0)

    # Pass the in-memory file to preprocess_audio
    processed_audio = preprocess_audio(buffer)

    # Check the shape of the processed audio
    assert processed_audio.shape == (1, 60), f"Unexpected shape: {processed_audio.shape}"

def test_get_emotion_label():
    """
    Test the get_emotion_label function for correct label mapping.
    """
    labels = ["Neutral", "Happy", "Sad", "Angry", "Fearful"]
    for i, label in enumerate(labels):
        assert get_emotion_label(i) == label, f"Expected {label}, got {get_emotion_label(i)}"
