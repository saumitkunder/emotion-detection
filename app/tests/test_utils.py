import numpy as np
from app.utils import preprocess_audio, get_emotion_label

def test_preprocess_audio():
    # Simulate an audio file using a NumPy array
    fake_audio = np.random.random(22050 * 5)  # 5 seconds of fake audio
    processed_audio = preprocess_audio(fake_audio)
    assert processed_audio.shape == (1, 60, fake_audio.shape[0] // 60)

def test_get_emotion_label():
    labels = ["Happy", "Sad", "Angry", "Fearful", "Neutral"]
    for i, label in enumerate(labels):
        assert get_emotion_label(i) == label
