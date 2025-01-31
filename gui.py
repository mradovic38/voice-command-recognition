from transcription_postprocessor import TranscriptionPostprocessor

import streamlit as st
from audio_recorder_streamlit import audio_recorder
import librosa
import io
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import os
import base64
import tqdm
from huggingface_hub import hf_hub_download, HfApi
from dtw import OOVHandler



class GUI():
    def __init__(self, model_name:str, postprocessor:TranscriptionPostprocessor, oov_handler:OOVHandler=None, 
                 model_dir:str='model', resources_dir:str='resources', use_cache:bool=False)-> None:
        self.model_name = model_name
        self.model_dir=model_dir
        self.resources = resources_dir
        self.postprocessor = postprocessor
        self.oov_handler = oov_handler

        self.processor = None
        self.model = None
        path = model_name
        if use_cache:
            self._download_model_if_not_present()
            path = model_dir
        self._load_model(path)

    def _download_model_if_not_present(self):
        """Downloads the model from Hugging Face if it's not already present in the project directory."""
        # Ensure the directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Check if model files are already present
        model_file_path = os.path.join(self.model_dir, "pytorch_model.bin")
        processor_file_path = os.path.join(self.model_dir, "config.json")
        
        # If model files are not present, download the model from Hugging Face
        if not os.path.exists(model_file_path) or not os.path.exists(processor_file_path):
            # Display download message and then clear it after the download is complete
            download_message = st.empty()
            download_message.write(f"Model not found in the local directory, downloading {self.model_name}...")
            
            # Add a progress bar for downloading
            progress_bar = st.progress(0)
            
            try:
                # Download the full model and all its files from Hugging Face
                self._download_full_model_from_huggingface(progress_bar)
                st.success(f"Model {self.model_name} downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading the model: {str(e)}")
                raise
            finally:
                # Remove the message and progress bar after download is finished
                download_message.empty()
                progress_bar.empty()

    def _download_full_model_from_huggingface(self, progress_bar):
        """Downloads all files from Hugging Face model repository to the local directory."""
        try:
            # Initialize Hugging Face API to get the model's file list
            api = HfApi()
            model_files = api.list_repo_files(self.model_name)

            total_files = len(model_files)
            for idx, file_name in enumerate(model_files):
                file_path = hf_hub_download(repo_id=self.model_name, filename=file_name, local_dir=self.model_dir)
                # Update progress bar
                progress_bar.progress(int((idx + 1) / total_files * 100))
                tqdm.tqdm.write(f"Downloaded {file_name} to {file_path}")

        except Exception as e:
            st.error(f"Error during model file download: {str(e)}")
            raise

    def _load_model(self, path:str):
        """Loads the model from the local directory."""
        try:
            # Load the processor and model
            self.processor = Wav2Vec2Processor.from_pretrained(path)
            self.model = Wav2Vec2ForCTC.from_pretrained(path)
        except Exception as e:
            st.error(f"Error loading model from local directory: {str(e)}")
            raise

    def run(self):
        
        st.title("Smart Home Voice Control")
        
        audio_bytes = audio_recorder(pause_threshold=2, sample_rate=16000)
        if audio_bytes:
            self._on_record_button_press(audio_bytes)
        # Initialize session state
        if 'system_state' not in st.session_state:
            st.session_state.system_state = {
                'light': False,
                'sound': False,
                'door': False,
                'door_selected': False,
                'sound_selected': False,
                'light_selected': False,
            }
            self._update_status_labels()

        
            
        
       

    def _process_command(self, command):
        # Access system state from session_state
        system_state = st.session_state.system_state
        match command:
            case "vrata":
                system_state['door_selected'] = True
                system_state['light_selected'] = False
                system_state['sound_selected'] = False
            case "svetlo":
                system_state['light_selected'] = True
                system_state['door_selected'] = False
                system_state['sound_selected'] = False
            case "zvuk":
                system_state['sound_selected'] = True
                system_state['light_selected'] = False
                system_state['door_selected'] = False
            case "otvori":
                if system_state['door_selected']:
                    system_state['door'] = True
            case "zatvori":
                if system_state['door_selected']:
                    system_state['door'] = False
            case "uključi":
                if system_state['light_selected']:
                    system_state['light'] = True
                elif system_state['sound_selected']:
                    system_state['sound'] = True
            case "isključi":
                if system_state['light_selected']:
                    system_state['light'] = False
                elif system_state['sound_selected']:
                    system_state['sound'] = False

    

    def _create_select_image(self, image_file):
        # Read image file and encode it in base64
        with open(os.path.join(self.resources, image_file), "rb") as image:
            encoded_image = base64.b64encode(image.read()).decode('utf-8')
        
        # HTML to display the image with a red border
        image_html = f'<div style="border: 2px solid red; display: inline-block;">'
        image_html += f'<img src="data:image/png;base64,{encoded_image}" style="width: 100px; height: 150px; object-fit: contain;"></div>'

        return image_html

    def _create_regular_image(self, image_file):
        with open(os.path.join(self.resources, image_file), "rb") as image:
            encoded_image = base64.b64encode(image.read()).decode('utf-8')

        image_html = f'<img src="data:image/png;base64,{encoded_image}" style="width: 100px; height: 150px; object-fit: contain;"></div>'

        return image_html

    def _update_status_labels(self):
        # Access system state from session_state
        system_state = st.session_state.system_state

        
        # Create placeholders for images and labels in the same row (using st.empty())
        col1, col2, col3 = st.columns(3)

        # Use st.empty() to keep placeholders to update the content dynamically
        with col1:
            light_image = st.empty()
            light_label = st.empty()

            light_label.write(f"Light: {'ON' if system_state['light'] else 'OFF'} {'SELECTED' if system_state['light_selected'] else ''}")
            
            image_file = "light on.png" if system_state['light'] else "light off.png"
            if system_state['light_selected']:     
                image_html = self._create_select_image(image_file)
            else:
                image_html = self._create_regular_image(image_file)

            light_image.markdown(image_html, unsafe_allow_html=True)

        with col2:
            sound_image = st.empty()
            sound_label = st.empty()

            sound_label.write(f"Sound: {'ON' if system_state['sound'] else 'OFF'} {'SELECTED' if system_state['sound_selected'] else ''}")

            image_file = "sound on.png" if system_state['sound'] else "sound off.png"
            if system_state['sound_selected']:     
                image_html = self._create_select_image(image_file)
            else:
                image_html = self._create_regular_image(image_file)

            sound_image.markdown(image_html, unsafe_allow_html=True)
            

        with col3:
            door_image = st.empty()
            door_label = st.empty()

            door_label.write(f"Door: {'OPEN' if system_state['door'] else 'CLOSED'} {'SELECTED' if system_state['door_selected'] else ''}")

            image_file = "door open.png" if system_state['door'] else "door shut.png"
            if system_state['door_selected']:     
                image_html = self._create_select_image(image_file)
            else:
                image_html = self._create_regular_image(image_file)

            door_image.markdown(image_html, unsafe_allow_html=True)

    def _predict_word(self, audio):
        try:
            input_values = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values

            with torch.no_grad():
                logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0].strip().lower()
            transcription = self.postprocessor.postprocess_transcription(transcription)
            return transcription
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
            return None

    def _on_record_button_press(self, audio_data):
        try:
            audio, _ = librosa.load(
                io.BytesIO(audio_data),  # Convert bytes to a file-like object
                sr=16000,               # Resample to 16k sample rate
                duration=2
            )
            oov_check = True
            if self.oov_handler:
                oov_check = self.oov_handler.check_if_oov(audio)

            if oov_check:
                transcription = self._predict_word(audio)
                if not transcription:
                    transcription = 'No trigger word detected.'
            else:
                transcription = 'No trigger word detected.'

            if transcription:
                self._process_command(transcription)
                
            self._update_status_labels()
        except:
            transcription = 'No trigger word detected.'
        

        st.write(f"Transcription: {transcription}")

