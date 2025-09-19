import json
import os
import numpy as np


class TextDataProcessor:
    """Process text data for character-level modeling"""

    def __init__(self, filepath, max_length):
        """
        Initialize the data processor

        Arguments:
        text_data -- raw text data (string or list of strings)
        """
        self.max_length = max_length
        self.filepath = filepath
        with open(filepath, 'r', encoding='utf-8') as f:
            self.words = [line.strip().lower() for line in f if line.strip()]

        print(f"Loaded {len(self.words)} names")
        print(f"Sample names: {self.words[:5]}")

        all_chars = set(''.join(self.words))

        all_chars.add('^')  # Start token
        all_chars.add('$')  # End token
        all_chars.add(' ')  # Space for padding

        self.chars = sorted(list(all_chars))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.n_chars = len(self.chars)

    def preprocess_training_data(self):
        """
        Returns:
        --------
        X, y : numpy arrays
            Training data and labels
        """

        print(f"Vocabulary size: {self.n_chars}")
        print(f"Characters: {''.join(self.chars)}")

        X, y = [], []

        for name in self.words:
            name_with_tokens = '^' + name + '$'

            for i in range(1, len(name_with_tokens)):
                input_seq = name_with_tokens[:i]  # Input: all characters up to position i
                output_char = name_with_tokens[i] # Output: next character

                if len(input_seq) < self.max_length: # Pad input sequence to max_length
                    input_seq = input_seq + ' ' * (self.max_length - len(input_seq))
                elif len(input_seq) > self.max_length:
                    input_seq = input_seq[-self.max_length:]
                # Convert to indices
                X.append([self.char_to_idx[c] for c in input_seq])
                y.append(self.char_to_idx[output_char])

        return np.array(X), np.array(y)


    def save(self, path_to_safe_dir):
        metadata = {
            'max_length': self.max_length,
            'filepath':self.filepath,
        }
        metadata_path = os.path.join(path_to_safe_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Metadata saved to {metadata_path}")


    @classmethod
    def load(cls, path_to_metadata):
        with open(path_to_metadata, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return cls(max_length = metadata['max_length'],filepath=metadata['filepath'])
