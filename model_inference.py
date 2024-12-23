import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa

def recognize_command(audio_chunk):
    """Recognize command from audio chunk."""
    # Preprocess audio (ensure proper tensor format)
    input_values = processor(audio_chunk, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    # Decode the predicted tokens and return the transcription
    return processor.decode(predicted_ids[0])

if __name__ == '__main__':
    # Load processor and fine-tuned model
    processor = Wav2Vec2Processor.from_pretrained("mradovic38/wav2vec2-large-xlsr-53-serbian-smart-home-commands")
    model = Wav2Vec2ForCTC.from_pretrained("mradovic38/wav2vec2-large-xlsr-53-serbian-smart-home-commands")
    model.eval()

    # Load the audio file using librosa (ensuring correct sampling rate)
    y, _ = librosa.load('vrata_test.wav', sr=16000)

    # Call the recognize_command function with the audio
    transcription = recognize_command(y)

    # Print the transcription
    print("Transcription:", transcription)
