from feature_extraction import extract_features
from dtw import calculate_dtw_cost

import os
import librosa
import numpy as np

class OOVHandler():
    def __init__(self, dataset_dir:str='dataset', threshold:float=.15, sr:int=16000) -> None:
        self.sr = sr
        self.threshold = threshold

        sounds = []
        
        for file_name in os.listdir(dataset_dir):
            if file_name.endswith(".wav"):
                file_path = os.path.join(dataset_dir, file_name)
                sounds.append(file_path)

        self.sounds = sounds

                
    def check_if_oov(self, y:np.ndarray) -> bool:
        y_features = extract_features(y, self.sr)
        min_dtw = np.inf

        for s in self.sounds:
            s_y, _ = librosa.load(s, sr=self.sr)

            s_features = extract_features(s_y, self.sr)

            cur_dtw = calculate_dtw_cost(y_features, s_features)

            min_dtw = min(cur_dtw, min_dtw)


        if min_dtw > self.threshold:
            return False
        return True


