import random
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import json

@keras.saving.register_keras_serializable()
def loss_function(real, pred):
    """Custom loss function with masking"""
    loss = keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)
    return tf.reduce_mean(loss)

class CharacterLevelRNN:
    """Character-level RNN for text generation using TensorFlow/Keras"""

    def __init__(self, vocab_size, stop_char, embedding_dim=256, rnn_units=1024, model_type='lstm'):
        """
        Initialize the character-level RNN model

        Arguments:
        vocab_size -- number of unique characters in vocabulary
        embedding_dim -- dimension of character embeddings
        rnn_units -- number of RNN units
        model_type -- type of RNN ('lstm', 'gru', or 'simple')
        """
        self.vocab_size = vocab_size
        self.stop_char = stop_char
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.model_type = model_type
        self.model = None

    def build_model(self, batch_size=None):
        """
        Build the RNN model architecture

        Arguments:
        batch_size -- batch size for training (None for variable batch size)

        Returns:
        model -- Keras model
        """
        if batch_size:
            model = keras.Sequential()
            model.add(layers.Input(batch_shape=(batch_size, None)))
            model.add(layers.Embedding(
                self.vocab_size,
                self.embedding_dim
            ))
        else:
            model = keras.Sequential()
            model.add(layers.Embedding(
                self.vocab_size,
                self.embedding_dim
            ))

        # RNN layers
        if self.model_type == 'lstm':
            model.add(layers.LSTM(
                self.rnn_units,
                return_sequences=True,
                stateful=True if batch_size else False,
                recurrent_initializer='glorot_uniform'
            ))
            model.add(layers.LSTM(
                self.rnn_units,
                return_sequences=True,
                stateful=True if batch_size else False,
                recurrent_initializer='glorot_uniform'
            ))
        elif self.model_type == 'gru':
            model.add(layers.GRU(
                self.rnn_units,
                return_sequences=True,
                stateful=True if batch_size else False,
                recurrent_initializer='glorot_uniform'
            ))
        else:
            model.add(layers.SimpleRNN(
                self.rnn_units,
                return_sequences=True,
                stateful=True if batch_size else False,
                recurrent_initializer='glorot_uniform'
            ))

        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(self.vocab_size))

        self.model = model
        return model

    def compile_model(self, learning_rate=0.01):
        """Compile the model with optimizer and loss function"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_function
        )

    def generate_text(self, start_string, char_to_idx, idx_to_char,
                      num_generate=100, temperature=1.0):
        """
        Generate text using the trained model

        Arguments:
        start_string -- seed text to start generation
        char_to_idx -- dictionary mapping characters to indices
        idx_to_char -- dictionary mapping indices to characters
        num_generate -- number of characters to generate
        temperature -- randomness in generation (0.1=deterministic, 2.0=very random)

        Returns:
        generated_text -- generated string
        """
        input_eval = [char_to_idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []

        if self.model.layers[1].stateful:
            self.model.reset_states()

        for _ in range(num_generate):
            predictions = self.model(input_eval)
            predictions = tf.squeeze(predictions, 0)

            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx_to_char[predicted_id])
            if idx_to_char[predicted_id] == self.stop_char:
                break

        return start_string + ''.join(text_generated)

    def save_model(self, filepath='char_rnn_model'):
        """
        Save the model architecture and weights

        Arguments:
        filepath -- path to save the model (without extension)
        """
        if self.model is None:
            raise ValueError("No model to save. Build and train a model first.")

        self.model.save(f'{filepath}.keras')

        config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'stop_char': self.stop_char,
            'rnn_units': self.rnn_units,
            'model_type': self.model_type
        }

        import json
        with open(f'{filepath}_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {filepath}.keras")
        print(f"Weights saved to {filepath}.weights.h5")
        print(f"Config saved to {filepath}_config.json")

    @classmethod
    def load_model(cls, filepath='char_rnn_model', weights_only=False):
        """
        Load a saved model

        Arguments:
        filepath -- path to the saved model (without extension)
        weights_only -- if True, only load weights (requires rebuilding architecture)

        Returns:
        model_instance -- CharacterLevelRNN instance with loaded model
        """

        with open(f'{filepath}_config.json', 'r') as f:
            config = json.load(f)

        instance = cls(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            rnn_units=config['rnn_units'],
            stop_char=config['stop_char'],
            model_type=config['model_type']
        )

        if weights_only:
            instance.build_model()
            instance.model.load_weights(f'{filepath}_weights.h5')
            print(f"Weights loaded from {filepath}_weights.h5")
        else:
            instance.model = keras.models.load_model(
                f'{filepath}.keras',
                custom_objects={'loss_function': loss_function}
            )
            print(f"Model loaded from {filepath}.keras")

        return instance

    def train_model(self, processor_, text_data, num_iterations=10000,
                    seq_length=10, print_every=2000):
        """
        Simple training function similar to the original implementation

        Arguments:
        processor_ -- Text data processor
        text_data -- list of training examples (e.g., names)
        num_iterations -- number of training iterations
        seq_length -- maximum sequence length
        print_every -- print loss and sample every N iterations

        Returns:
        model -- trained model
        processor -- data processor with vocabulary mappings
        """

        model = self.model
        processor = processor_

        examples = [example.strip() for example in text_data]
        losses = []

        for iteration in range(num_iterations):
            idx = iteration % len(examples)
            example = examples[idx]

            if len(example) > 0:
                input_seq = processor.vocab[0] + example
                target_seq = example + processor.stop_char

                input_ids = [processor.char_to_idx.get(c, 0) for c in input_seq]
                target_ids = [processor.char_to_idx.get(c, 0) for c in target_seq]

                max_len = max(len(input_ids), len(target_ids))
                input_ids = input_ids[:max_len]
                target_ids = target_ids[:max_len]

                if len(input_ids) < seq_length:
                    input_ids = input_ids + [0] * (seq_length - len(input_ids))
                    target_ids = target_ids + [0] * (seq_length - len(target_ids))

                input_tensor = tf.expand_dims(input_ids[:seq_length], 0)
                target_tensor = tf.expand_dims(target_ids[:seq_length], 0)

                with tf.GradientTape() as tape:
                    predictions = model(input_tensor, training=True)
                    loss = loss_function(target_tensor, predictions)

                gradients = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                losses.append(loss.numpy())

            if iteration % print_every == 0:
                avg_loss = np.mean(losses[-100:]) if losses else 0
                print(f'Iteration: {iteration}, Loss: {avg_loss:.4f}\n')

                for i in range(3):
                    start_char = random.choice(processor.vocab[1:])
                    generated = self.generate_text(
                        start_char,
                        processor.char_to_idx,
                        processor.idx_to_char,
                        num_generate=10,
                        temperature=0.05
                    )
                    print(generated)
                print('\n')

        self.model = model
        return self, processor
