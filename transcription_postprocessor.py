from fuzzywuzzy import fuzz
from typing import List

class TranscriptionPostprocessor():
    def __init__(self, similarity_threshold:float=70,
                 valid_commands:List[str]=["uključi", "isključi", "svetlo", "zvuk", "otvori", "zatvori", "vrata"]) -> None:
        self.valid_commands = valid_commands
        self.similarity_threshold = similarity_threshold

    def postprocess_transcription(self, transcription):
        """
        Postprocesses the transcription to match one of the predefined commands based on similarity.
        If the transcription is similar enough to one of the commands, returns the matched command.
        If not, returns None.
        
        Args:
        - transcription (str): The raw transcription string from the speech-to-text model.
        
        Returns:
        - str or None: The matched command or None if no valid match is found.
        """
        # Normalize transcription by removing extra spaces and converting to lowercase
        transcription = transcription.lower().strip()

        best_match = None
        highest_score = 0
        
        # Compare transcription with each valid command
        for command in self.valid_commands:
            score = fuzz.ratio(transcription, command)  # Calculate similarity score
            if score > highest_score:
                highest_score = score
                best_match = command
        
        # If the highest similarity score is above the threshold, return the best match
        if highest_score >= self.similarity_threshold:
            return best_match
        else:
            return None  # Return None if no valid match is found

    # # Example usage:
    # transcription = "ukluci svetlo"  # Example transcription from Wav2Vec2
    # matched_command = postprocess_transcription(transcription)

    # if matched_command:
    #     print(f"Matched Command: {matched_command}")
    # else:
    #     print("No valid match found")
