
import librosa
import numpy as np

def extract_features(y: np.ndarray, sr:int=16000) -> np.ndarray:
    """
    Extract audio features from a file using MFCC.
    
    Args:
        file_path (str): Path to the audio file
    
    Returns:
        numpy.ndarray: Extracted features
    """
    y[y == 0] = 1e-10
    y = librosa.util.normalize(y)

    # MFCC extraction with normalization
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    features = mfccs.T
    
    return features