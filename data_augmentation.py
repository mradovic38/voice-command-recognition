import streamlit as st
import sounddevice as sd
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import time
import matplotlib.pyplot as plt

# Load the model and processor from HuggingFace
@st.cache_resource
def load_model():
    try:
        model_name = "mradovic38/wav2vec2-large-xlsr-53-serbian-smart-home-commands"
        processor = Wav2Vec2Processor.from_pretrained(
            model_name,
            local_files_only=False,
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            local_files_only=False,
        )
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None  # Return None for both processor and model in case of an error

# Initialize model
processor, model = load_model()

# Check if model was loaded successfully
if processor is None or model is None:
    st.error("Failed to load the model. Exiting...")
    st.stop()  # Stop the execution if the model failed to load

# Set up the Streamlit page
st.title("Smart Home Voice Control")

# Initialize session state
if 'system_state' not in st.session_state:
    st.session_state.system_state = {
        'light': False,
        'sound': False,
        'door': False,
        'last_command': None,
        'last_command_time': time.time()
    }

# Create columns for status display
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Light")
    if st.session_state.system_state['light']:
        st.success("ON")
    else:
        st.error("OFF")

with col2:
    st.subheader("Sound")
    if st.session_state.system_state['sound']:
        st.success("ON")
    else:
        st.error("OFF")

with col3:
    st.subheader("Door")
    if st.session_state.system_state['door']:
        st.success("OPEN")
    else:
        st.error("CLOSED")

# Display last command
st.subheader("Last Command")
if st.session_state.system_state['last_command']:
    st.info(st.session_state.system_state['last_command'])
else:
    st.info("Waiting for command...")

def process_command(command):
    """Process the recognized command and update system state"""
    if time.time() - st.session_state.system_state['last_command_time'] < 1:
        return  # Ignore commands that come too quickly

    command = command.lower().strip()
    st.session_state.system_state['last_command'] = command
    st.session_state.system_state['last_command_time'] = time.time()

    if "svetlo" in command:
        if "uključi" in command:
            st.session_state.system_state['light'] = True
        elif "isključi" in command:
            st.session_state.system_state['light'] = False
    elif "zvuk" in command:
        if "uključi" in command:
            st.session_state.system_state['sound'] = True
        elif "isključi" in command:
            st.session_state.system_state['sound'] = False
    elif "vrata" in command:
        if "otvori" in command:
            st.session_state.system_state['door'] = True
        elif "zatvori" in command:
            st.session_state.system_state['door'] = False

def predict_word(audio):
    """Predict word from audio"""
    try:
        input_values = processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding=True
        ).input_values
        
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def record_audio(duration=3, fs=16000):
    """Record audio for a short duration"""
    audio = sd.rec(int(fs * duration), samplerate=fs, channels=1)
    sd.wait()
    return audio.flatten().astype(np.float32)

# Button to start recording
record_button = st.button("Press and Hold to Speak Command")

# Create an empty container for the waveform plot
waveform_container = st.empty()

# Record audio only when the button is pressed
if record_button:
    st.write("Listening for command...")
    
    # Create an empty plot
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_title("Waveform of Audio")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")
    
    # Real-time waveform display during recording
    audio_data = []
    def audio_callback(indata, frames, time, status):
        """Callback function to get audio data in real-time"""
        audio_data.extend(indata[:, 0])  # Collect the audio data
        ax.clear()
        ax.plot(audio_data)
        ax.set_title("Waveform of Audio")
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude")
        waveform_container.pyplot(fig)  # Update the plot in Streamlit

    # Start recording and show waveform in real-time
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
        sd.sleep(3000)  # Listen for 3 seconds or until the button is released

    # After recording, process the audio
    if len(audio_data) > 0:
        transcription = predict_word(np.array(audio_data))
        if transcription:
            st.write(f"Transcription: {transcription}")  # Display transcription for debugging
            process_command(transcription)
