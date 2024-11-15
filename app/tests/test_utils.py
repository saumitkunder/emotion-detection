import pytest
import numpy as np
import tempfile
import wave
from app.utils import preprocess_audio

def test_preprocess_audio():
    """
    Test the audio preprocessing function.
    """
    # Generate dummy audio data (3 seconds of random audio)
    sample_rate = 22050
    duration = 3
    dummy_audio = np.random.uniform(-1, 1, sample_rate * duration)

    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        with wave.open(temp_audio_file.name, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono audio
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((dummy_audio * 32767).astype(np.int16).tobytes())

        # Test the preprocessing function with the temporary file
        try:
            features = preprocess_audio(temp_audio_file.name)
            assert features is not None, "Features should not be None"
            assert features.shape == (1, 60), f"Expected (1, 60), got {features.shape}"
        except ValueError as ve:
            pytest.fail(f"Preprocessing failed: {ve}")


# def test_get_emotion_label():
#     """
#     Test the get_emotion_label function.
#     """
#     labels = ["Neutral", "Happy", "Sad", "Angry", "Fearful"]
#     for i, label in enumerate(labels):
#         assert get_emotion_label(i) == label, f"Expected {label}, got {get_emotion_label(i)}"
#     with pytest.raises(ValueError):
#         get_emotion_label(-1)
#     with pytest.raises(ValueError):
#         get_emotion_label(100)
