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
    # Ensure there are no zero values to avoid log of zero errors
    y[y == 0] = 1e-10
    
    # Normalize audio signal
    y = librosa.util.normalize(y)
    
    
    # MFCC extraction
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Normalize MFCCs individually per feature
    mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(mfccs, axis=1, keepdims=True)
    
    
    return mfccs.T  # Transpose to match the expected (time x feature) format