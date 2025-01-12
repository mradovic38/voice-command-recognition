import librosa
import numpy as np

def extract_features(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract audio features from a file using MFCC
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sample rate (default: 16000)
    
    Returns:
        np.ndarray: Extracted features (MFCC)
    """
    # Pre-emphasis filter to emphasize higher frequencies
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # Normalize audio signal to have zero mean and unit variance
    y = librosa.util.normalize(y)

    # Frame size and hop size
    frame_size = int(0.025 * sr)  # 25 ms
    hop_size = int(0.010 * sr)   # 10 ms

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=hop_size, window='hamming'
    )

    # Compute Delta and Delta-Delta features
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    # Combine static, delta, and delta-delta features
    combined_features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])

    # Normalize the combined features (zero mean and unit variance)
    combined_features = (combined_features - np.mean(combined_features)) / np.std(combined_features)

    return combined_features.T  # Transpose to (time x feature) format