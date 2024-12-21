from data_augmentation import AudioAugmentation

import torch
import librosa
from transformers import Wav2Vec2Processor
import numpy as np

class Preprocessor():
    def __init__(self, processor:Wav2Vec2Processor, sr:int=16000, audio_augmentation:AudioAugmentation=None, 
                 augment_count:int=2) -> None:
        self.processor = processor
        self.sr = sr
        self.audio_augmentation = audio_augmentation
        self.augment_count = augment_count

    def preprocess(self, row):
        # Load audio with librosa
        audio_input, _ = librosa.load(row["audio_filepath"], sr=self.sr)  # Resample to 16 kHz

        # Prepare list to store audio inputs and labels
        preprocessed_data = []

        # Add the original audio
        inputs = self.processor(audio_input, sampling_rate=self.sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values[0]
        labels = self.processor.tokenizer(row["text"]).input_ids
        preprocessed_data.append({"input_values": input_values, "labels": torch.tensor(labels)})

        # Generate augmentations
        if self.audio_augmentation:
            for _ in range(self.augment_count):
                augmented_audio = self.audio_augmentation.augment_audio(audio_input, self.sr)
                inputs = self.processor(augmented_audio, sampling_rate=self.sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values[0]
                preprocessed_data.append({"input_values": input_values, "labels": torch.tensor(labels)})

        return preprocessed_data