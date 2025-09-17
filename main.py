import json
import pickle

from super_model import *

stop_char = ','

class TextDataProcessor:
    """Process text data for character-level modeling"""

    def __init__(self, text_data):
        """
        Initialize the data processor

        Arguments:
        text_data -- raw text data (string or list of strings)
        """

        self.text = text_data
        self.stop_char = stop_char
        self.vocab = sorted(set(self.text))
        self.vocab_size = len(self.vocab)

        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        self.text_as_int = np.array([self.char_to_idx[c] for c in self.text])

    def save_mappings(self, filepath='vocab_mappings'):
        """
        Save vocabulary mappings to files

        Arguments:
        filepath -- base filepath for saving mappings
        """
        mappings = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()}
        }

        with open(f'{filepath}.json', 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)

        with open(f'{filepath}.pkl', 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'vocab_size': self.vocab_size,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char
            }, f)

        print(f"Vocabulary mappings saved to {filepath}.json and {filepath}.pkl")

    @classmethod
    def load_mappings(cls, filepath='vocab_mappings', format='json'):
        """
        Load vocabulary mappings from files

        Arguments:
        filepath -- base filepath for loading mappings
        format -- 'json' or 'pickle'

        Returns:
        processor -- TextDataProcessor instance with loaded mappings
        """
        if format == 'json':
            with open(f'{filepath}.json', 'r', encoding='utf-8') as f:
                mappings = json.load(f)

            mappings['idx_to_char'] = {int(k): v for k, v in mappings['idx_to_char'].items()}
        else:
            with open(f'{filepath}.pkl', 'rb') as f:
                mappings = pickle.load(f)

        processor = cls.__new__(cls)
        processor.vocab = mappings['vocab']
        processor.vocab_size = mappings['vocab_size']
        processor.char_to_idx = mappings['char_to_idx']
        processor.idx_to_char = mappings['idx_to_char']
        processor.text = None
        processor.text_as_int = None

        print(f"Vocabulary mappings loaded from {filepath}.{format}")
        return processor


def save_complete_model(char_rnn, processor, base_path='complete_model'):
    """
    Save everything needed to use the model later

    Arguments:
    model -- trained Keras model
    char_rnn -- CharacterLevelRNN instance
    processor -- TextDataProcessor instance
    base_path -- base path for all saved files
    """
    import os

    os.makedirs(base_path, exist_ok=True)
    model_path = os.path.join(base_path, 'model')
    char_rnn.save_model(model_path)
    vocab_path = os.path.join(base_path, 'vocab')
    processor.save_mappings(vocab_path)

    print(f"\nComplete model saved to '{base_path}/' directory")
    print("Files created:")
    print(f"  - model.keras (full model)")
    print(f"  - model_weights.h5 (weights only)")
    print(f"  - model_config.json (architecture config)")
    print(f"  - vocab.json (vocabulary mappings)")
    print(f"  - vocab.pkl (vocabulary mappings - pickle format)")

def load_complete_model(base_path='complete_model', weights_only=False):
    """
    Load everything needed to use the model

    Arguments:
    base_path -- base path where model files are stored
    weights_only -- if True, only load weights (smaller, faster)

    Returns:
    char_rnn -- CharacterLevelRNN instance with loaded model
    processor -- TextDataProcessor with vocabulary mappings
    """
    import os

    model_path = os.path.join(base_path, 'model')
    char_rnn = CharacterLevelRNN.load_model(model_path, weights_only=weights_only)

    vocab_path = os.path.join(base_path, 'vocab')
    processor = TextDataProcessor.load_mappings(vocab_path)

    print(f"\nComplete model loaded from '{base_path}/' directory")
    return char_rnn, processor

def create_model(text_):
    new_processor = TextDataProcessor(text_)
    new_model = CharacterLevelRNN(
        vocab_size=new_processor.vocab_size,
        stop_char=new_processor.stop_char,
        embedding_dim=64,
        rnn_units=128,
        model_type='lstm'
    )
    new_model.build_model()
    new_model.compile_model(learning_rate=0.01)
    return new_model, new_processor


if __name__ == "__main__":
    print("Training Character-Level RNN with TensorFlow")
    print("=" * 50)

    text = open('names.txt', 'r').read().lower().replace(' ', '')

    loaded_rnn, loaded_processor = create_model(text)

    names = [ w + loaded_processor.stop_char for w in text.split(stop_char)]

    loaded_rnn.train_model(
        loaded_processor,
        names,
        num_iterations=10000,
        print_every=500
    )

    # SAVE THE MODEL
    print("\n" + "=" * 50)
    print("SAVING THE MODEL")
    print("=" * 50)
    save_complete_model(loaded_rnn, loaded_processor, 'super_model')

    print("\n" + "=" * 50)
    print("LOADING AND USING SAVED MODEL")
    print("=" * 50)

    loaded_rnn, loaded_processor = load_complete_model('super_model')
    loaded_rnn.model.summary()

    print(
        loaded_rnn.generate_text(
            start_string="abba",
            char_to_idx=loaded_processor.char_to_idx,
            idx_to_char= loaded_processor.idx_to_char,
            num_generate=5,
            temperature=0.05
        )
    )

    # print("\n" + "=" * 50)
    # print("DIFFERENT TEMPERATURE SETTINGS")
    # print("=" * 50)

    # Show effect of temperature
    # temps = [0.5, 0.8, 1.0, 1.5]
    # loaded_rnn, loaded_processor = load_complete_model('super_model')
    #
    # for temp in temps:
    #     print(f"\nTemperature = {temp} (lower=more conservative, higher=more random):")
    #     for _ in range(3):
    #         generated = loaded_rnn.generate_text(
    #             'a',
    #             loaded_processor.char_to_idx,
    #             loaded_processor.idx_to_char,
    #             temperature=temp
    #         )
    #         print(f"  - {generated.split()[0] if generated else 'Empty'}")
