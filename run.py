from transcription_postprocessor import TranscriptionPostprocessor
from gui import GUI

from oov_check import OOVHandler


if __name__ =='__main__':
    postprocessor = TranscriptionPostprocessor()
    oov_handler = OOVHandler('dataset', threshold=0.45, sr=16000, username_if_in_dataset='38-21')

    smart_gui = GUI(model_name="mradovic38/wav2vec2-large-xlsr-53-serbian-smart-home-commands",
                    postprocessor=postprocessor,
                    oov_handler=oov_handler,
                    model_dir='models_cache',
                    resources_dir='resources')

    smart_gui.run()