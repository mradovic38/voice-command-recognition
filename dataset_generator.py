import os
import csv

from typing import Dict

class DatasetGenerator:

    def __init__(self, word_character_map:Dict[str, str]={}) -> None:
        self.word_character_map = word_character_map

    def generate(self, input_dir:str='dataset', output_file:str='data.csv') -> None:
        with open(output_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["audio_filepath", "text"])

            # Iterate through audio files
            for filename in os.listdir(input_dir):
                if filename.endswith(".wav"):

                    word = filename.split("-")[0]
                    word =  self.word_character_map[word] if word in self.word_character_map else word
                    audio_path = os.path.join(input_dir, filename)
                    writer.writerow([audio_path, word])

        print(f"Dataset saved to {output_file}")


if __name__ == '__main__':
    word_character_map = {
        'iskljuci': 'isključi',
        'ukljuci': 'uključi'
    }

    dg = DatasetGenerator(word_character_map)
    
    
    dg.generate(input_dir='dataset', output_file='data.csv')
    