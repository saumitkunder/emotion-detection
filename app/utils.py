import numpy as np
import librosa

def preprocess_audio(audio_file, sample_rate=22050, duration=3, n_mfcc=60):
    """
    Preprocesses the audio file for models expecting a flattened MFCC feature vector.
    """
    try:
        y, sr = librosa.load(audio_file, sr=sample_rate, mono=True, duration=duration)
        max_len = int(sample_rate * duration)
        if len(y) > max_len:
            y = y[:max_len]
        else:
            y = np.pad(y, (0, max_len - len(y)), mode="constant")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_flattened = np.mean(mfccs, axis=1).reshape(1, -1)
        return mfccs_flattened
    except Exception as e:
        raise ValueError(f"Error in preprocessing audio: {e}")

def get_emotion_label(index):
    """
    Maps the model's prediction index to an emotion label.
    """
    labels = ["Neutral", "Happy", "Sad", "Angry", "Fearful"]
    if index < 0 or index >= len(labels):
        raise ValueError("Index out of range for emotion labels")
    return labels[index]
