from transcription_postprocessor import TranscriptionPostprocessor
from gui import GUI

import os
import subprocess
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


if __name__ =='__main__':
    postprocessor = TranscriptionPostprocessor()
    smart_gui = GUI(model_name="mradovic38/wav2vec2-large-xlsr-53-serbian-smart-home-commands",
                    postprocessor=postprocessor,
                    model_dir='models_cache',
                    resources_dir='resources')

    smart_gui.run()