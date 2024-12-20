import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load processor and fine-tuned model
processor = Wav2Vec2Processor.from_pretrained("wav2vec2-serbian-commands")
model = Wav2Vec2ForCTC.from_pretrained("wav2vec2-serbian-commands")
model.eval()

def recognize_command(audio_chunk):
    """Recognize command from audio chunk."""
    # Preprocess audio
    input_values = processor(audio_chunk, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0])

