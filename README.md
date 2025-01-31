# Smart Home Voice Command Recognition


Smart home controller simulator, receiving voice commands from a microphone.\
Trained to detect the words: "vrata", "svetlo", "zvuk", "otvori", "zatvori", "uključi" and "isključi" to control the state of door, lights and audio in a smart home system.

## ❓ How to Run
### Online
Visit: https://smart-home-serbian-voice-controller.streamlit.app
    
### Locally
1. Clone the repository:
```bash
git clone https://github.com/mradovic38/voice-command-recognition
```
2. Run the following command in terminal:
```bash
streamlit run run.py
```

## 🤖 [Augmentation and Preprocessing](data_augmentation.py)

Since the dataset is relatively small, audio augmentation techniques were performed to expand the training dataset size. In this case the training dataset size was doubled. Three different augmentations were perfomed randomly:
  - Adding noise
  - Time Stretching
  - Pitch shifting
The augmentations were performed using the class [`AudioAugmentation`](data_augmentation.py), only on the training dataset to ensure valid evaluation.

## 🗣️ [Fine-Tuning Wav2Vec2](wav2vec2_fine_tuning.py)

Wav2Vec2 model for cross-lingual speech representations ([Wav2Vec2-XLSR-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)) was fine-tuned for this problem, since our smart home commands are in Serbian language.

To ensure proper evaluation, training examples and validation examples contain audio recordings of different speakers. If a speaker's voice is in both training and validation datasets, the validation would not correctly evaluate the model, resulting in poor performance.

The model was fine-tuned for 100 epochs with batch size of 8 since the dataset is relatively small. Increasing dropout yields better performance in this case as well, due to dataset size.

Fine-tuned model is available on Hugging Face 🤗 on the following link: \
[wav2vec2-large-xlsr-53-serbian-smart-home-commands](https://huggingface.co/mradovic38/wav2vec2-large-xlsr-53-serbian-smart-home-commands) 

<img src="https://github.com/user-attachments/assets/b672db3d-94e9-453c-854c-21745db885f3" width=70%>\
*Figure 1: Training loss over time.*

\
<img src="https://github.com/user-attachments/assets/fe5085c8-51ca-4b9a-83f2-b6e05229fdec" width=70%>\
*Figure 2: Validation loss over time.*

\
<img src="https://github.com/user-attachments/assets/cc6a42ef-002d-407c-b219-1914e232a663" width=70%>\
*Figure 2: Validation WER over time.*



## 🔇 [Out-Of-Vocabulary Detection](dtw/oov_check.py)

Since the dataset contains only the words, we do not have any way to detect words that are out of the vocabulary. That's why [`OOVHandler`](dtw/oov_check.py) class is introduced. Here the minimum distance from each of the words from the dataset is being calculated using Dynamic time warping (DTW). If that distance exceeds a given threshold, we label the word as out of the vocabulary (method `check_if_oov()` returns false). To perform DTW, we need to extract audio features. In this case, Mel-frequency cepstrum coefficients (MFCC) features were extracted, with delta and delta2 features was used.

## 🔠 [Postprocessing](transcription_postprocessor.py)

Sometimes, the model predicts the word that is very close to one of the words in the vocabulary (e.g. "uključi" is sometimes predicted as "uključii"). These close predictions should be mapped to the corresponding exact words. Class [`TranscriptionPostprocessor`](transcription_postprocessor.py) performs the mapping if the word is at least 70% near the word from the vocabulary.

## 💻 [GUI](gui.py)
GUI was created using streamlit. It captures a short audio recording of a command when the record button is clicked. If the user said one of the appropriate commands, the state of the images on the screen would change, simulating smart home voice control.

## 📖 Resources

  - [Facebook's XLSR-Wav2Vec2](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)
  - [Fine-Tune XLSR-Wav2Vec2 on Turkish ASR with 🤗 Transformers by Patrick von Platen](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_🤗_Transformers.ipynb#scrollTo=rrv65aj7G95i)
