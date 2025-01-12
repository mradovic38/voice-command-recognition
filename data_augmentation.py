import numpy as np
import librosa

class AudioAugmentation():

    def __init__(self, min_noise:float=0, max_noise:float=0.005, time_stretch_rate:float=0.9, pitch_shift_n_steps:int=2,
                 time_stretch_prob:float=.5, pitch_shift_prob:float=.5) -> None:
        self.min_noise=min_noise
        self.max_noise = max_noise
        self.time_stretch_rate = time_stretch_rate
        self.pitch_shift_n_steps = pitch_shift_n_steps
        self.time_stretch_prob = time_stretch_prob
        self.pitch_shift_prob = pitch_shift_prob

    def augment_audio(self, audio, sample_rate):
        # Add noise
        noise = np.random.normal(self.min_noise, self.max_noise, audio.shape)
        augmented_audio = audio + noise

        # Time stretching
        if np.random.rand() > self.time_stretch_prob:
            augmented_audio = librosa.effects.time_stretch(augmented_audio, rate=self.time_stretch_rate)

        # Pitch shift
        if np.random.rand() > self.pitch_shift_prob:
            augmented_audio = librosa.effects.pitch_shift(augmented_audio, sr=sample_rate, n_steps=self.pitch_shift_n_steps)

        return augmented_audio