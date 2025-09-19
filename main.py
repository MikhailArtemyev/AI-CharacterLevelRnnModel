import json
import pickle

from BiDir_SuperModel import *
from InputProcessor import *
#
# stop_char = ','
#
#
# def save_complete_model(char_rnn, processor, base_path='complete_model'):
#     """
#     Save everything needed to use the model later
#
#     Arguments:
#     model -- trained Keras model
#     char_rnn -- CharacterLevelRNN instance
#     processor -- TextDataProcessor instance
#     base_path -- base path for all saved files
#     """
#     import os
#
#     os.makedirs(base_path, exist_ok=True)
#     model_path = os.path.join(base_path, 'model')
#     char_rnn.save_model(model_path)
#     vocab_path = os.path.join(base_path, 'vocab')
#     processor.save_mappings(vocab_path)
#
#     print(f"\nComplete model saved to '{base_path}/' directory")
#     print("Files created:")
#     print(f"  - model.keras (full model)")
#     print(f"  - model_weights.h5 (weights only)")
#     print(f"  - model_config.json (architecture config)")
#     print(f"  - vocab.json (vocabulary mappings)")
#     print(f"  - vocab.pkl (vocabulary mappings - pickle format)")
#
# def load_complete_model(base_path='complete_model', weights_only=False):
#     """
#     Load everything needed to use the model
#
#     Arguments:
#     base_path -- base path where model files are stored
#     weights_only -- if True, only load weights (smaller, faster)
#
#     Returns:
#     char_rnn -- CharacterLevelRNN instance with loaded model
#     processor -- TextDataProcessor with vocabulary mappings
#     """
#     import os
#
#     model_path = os.path.join(base_path, 'model')
#     char_rnn = CharacterLevelRNN.load_model(model_path, weights_only=weights_only)
#
#     vocab_path = os.path.join(base_path, 'vocab')
#     processor = TextDataProcessor.load_mappings(vocab_path)
#
#     print(f"\nComplete model loaded from '{base_path}/' directory")
#     return char_rnn, processor
#
# def create_model(text_):
#     new_processor = TextDataProcessor(text_)
#     new_model = CharacterLevelRNN(
#         vocab_size=new_processor.vocab_size,
#         stop_char=new_processor.stop_char,
#         embedding_dim=64,
#         rnn_units=128,
#         model_type='lstm'
#     )
#     new_model.build_model()
#     new_model.compile_model(learning_rate=0.01)
#     return new_model, new_processor


if __name__ == "__main__":
    starting_letters = ['A', 'J', 'M', 'S', 'M', 'Ma', 'Mik', 'Si', 'Ba', 'Ta', 'Ke']
    max_length = 15

    # processor = TextDataProcessor("names.txt", max_length)
    # super_model = SuperRNN(
    #     text_processor=processor,
    #     gru_units=[128, 64],
    #     dropout_rate=0.3,
    #     learning_rate=0.001
    # )
    #
    # print("\nTraining model...")
    # history = super_model.train(
    #     epochs=1000,
    #     batch_size=32,
    #     validation_split=0.2
    # )
    #
    # super_model.save_model('super_model')

    super_model = SuperRNN.load_model("super_model")
    for l in starting_letters:
        print(super_model.generate(temperature=0.2, start_char=l))