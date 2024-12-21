import torch
import librosa
from transformers import Wav2Vec2Processor

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class Preprocessor():
    def __init__(self, processor:Wav2Vec2Processor, sr:int=16000) -> None:
        self.processor = processor
        self.sr = sr

    def preprocess(self, row):
        # Load audio with librosa
        audio_input, _ = librosa.load(row["audio_filepath"], sr=self.sr)  # Resample to 16 kHz
        # Process audio to input values
        inputs = self.processor(audio_input, sampling_rate=self.sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values[0]
        # Tokenize text
        labels = self.processor.tokenizer(row["text"]).input_ids
        return {"input_values": input_values, "labels": torch.tensor(labels)}