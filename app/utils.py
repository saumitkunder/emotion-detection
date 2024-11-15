import numpy as np
import librosa
import io

import numpy as np
import librosa
import io

def preprocess_audio(audio_file, sample_rate=22050, duration=3, n_mfcc=60):
    """
    Preprocesses the audio file for models expecting a flattened MFCC feature vector.
    Steps:
    - Converts audio to mono.
    - Resamples to a fixed sample rate.
    - Trims or pads to a fixed duration.
    - Extracts MFCCs.
    - Averages across time steps to flatten into a 1D feature vector.
    
    Args:
        audio_file: The uploaded audio file (e.g., from Streamlit).
        sample_rate: The target sample rate for audio processing.
        duration: The fixed duration (in seconds) for input audio.
        n_mfcc: The number of MFCC features to extract.

    Returns:
        A feature vector of shape (1, n_mfcc).
    """
    try:
        # Load the audio file
        y, sr = librosa.load(io.BytesIO(audio_file.read()), sr=sample_rate, mono=True, duration=duration)

        # Trim or pad to ensure fixed length
        max_len = int(sample_rate * duration)
        if len(y) > max_len:
            y = y[:max_len]
        else:
            y = np.pad(y, (0, max_len - len(y)), mode='constant')

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Average across time steps to reduce to a 1D feature vector
        mfccs_flattened = np.mean(mfccs, axis=1).reshape(1, -1)  # Shape: (1, n_mfcc)

        return mfccs_flattened
    except Exception as e:
        raise ValueError(f"Error in preprocessing audio: {e}")



def get_emotion_label(index):
    """
    Maps a model prediction index to a human-readable emotion label.
    
    Args:
        index: The index of the predicted emotion class (int).

    Returns:
        A string label corresponding to the emotion class.
    """
    emotion_labels = {
        0: "Happy",
        1: "Sad",
        2: "Angry",
        3: "Neutral",
        4: "Fearful",
        5: "Surprised",
        6: "Disgusted"
    }
    return emotion_labels.get(index, "Unknown")


def load_sample_audio(file_path, sample_rate=22050, duration=3, n_mfcc=60):
    """
    Loads and preprocesses a sample audio file for testing or debugging.
    
    Args:
        file_path: Path to the audio file on disk.
        sample_rate: Target sample rate for processing.
        duration: Fixed duration (in seconds) for the input audio.
        n_mfcc: Number of MFCC features to extract.

    Returns:
        A flattened feature vector ready for model input.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=sample_rate, mono=True, duration=duration)

        # Trim or pad audio to the fixed duration
        max_len = int(sample_rate * duration)
        if len(y) > max_len:
            y = y[:max_len]
        else:
            y = np.pad(y, (0, max_len - len(y)), mode="constant")

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Flatten MFCCs
        return mfccs.flatten().reshape(1, -1)  # Add batch dimension
    except Exception as e:
        raise ValueError(f"Error loading sample audio file: {e}")
