import os

import tensorflow as tf
import keras
from keras import layers
import numpy as np
from InputProcessor import TextDataProcessor

class SuperRNN:

    def __init__(
            self,
            text_processor,
            gru_units = None,
            dropout_rate=0.2,
            learning_rate=0.001,
            model = None
    ):
        """
        Create the bidirectional GRU model for name generation.

        Parameters:
        -----------
        gru_units : list
            Number of units for each GRU layer
        dropout_rate : float
            Dropout rate
        learning_rate : float
            Learning rate for optimizer

        Returns:
        --------
        model : keras.Model
            Compiled model
        """
        self.text_processor = text_processor
        if model is not None:
            self.model = model
            return
        # Input layer
        inputs = layers.Input(shape=(text_processor.max_length,))

        # Embedding layer
        x = layers.Embedding(
            input_dim=text_processor.n_chars,
            output_dim=32,
            input_length=text_processor.max_length
        )(inputs)

        # First bidirectional GRU layer
        x = layers.Bidirectional(
            layers.GRU(
                units=gru_units[0],
                return_sequences=True if len(gru_units) > 1 else False,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate * 0.5
            )
        )(x)

        # Additional bidirectional GRU layers
        for i, units in enumerate(gru_units[1:], start=1):
            return_sequences = i < len(gru_units) - 1
            x = layers.Bidirectional(
                layers.GRU(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate * 0.5
                )
            )(x)

            # Add batch normalization
            x = layers.BatchNormalization()(x)

        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)

        # Output layer - predict next character
        outputs = layers.Dense(text_processor.n_chars, activation='softmax')(x)

        # Create and compile model
        self.model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, epochs=50, batch_size=128, validation_split=0.2):
        """
        Train the model.

        Parameters:
        -----------
        X, y : numpy arrays
            Training data and labels
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        validation_split : float
            Fraction of data to use for validation

        Returns:
        --------
        history : keras.History
            Training history
        """
        X, y = self.text_processor.preprocess_training_data()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'super_model/super_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        return self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

    def generate(self, temperature=1.0, start_char=None):
        """
        Generate a new name using the trained model.

        Parameters:
        -----------
        temperature : float
            Controls randomness (0.1=conservative, 1.0=balanced, 2.0=creative)
        start_char : str
            Optional starting character for the name

        Returns:
        --------
        name : str
            Generated name
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if start_char and start_char in self.text_processor.char_to_idx:
            generated = '^' + start_char.lower()
        else:
            generated = '^'

        while len(generated) < self.text_processor.max_length:
            input_seq = generated + ' ' * (self.text_processor.max_length - len(generated))
            input_seq = input_seq[:self.text_processor.max_length]

            input_indices = np.array([[self.text_processor.char_to_idx[c] for c in input_seq]])

            predictions = self.model.predict(input_indices, verbose=0)[0]

            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))

            next_idx = np.random.choice(len(predictions), p=predictions)
            next_char = self.text_processor.idx_to_char[next_idx]

            if next_char == '$':
                break

            generated += next_char

        name = generated.replace('^', '').replace('$', '').strip()
        return name.capitalize()

    def save_model(self, path_to_safe_dir):
        model_path = os.path.join(path_to_safe_dir, 'super_model.keras')
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        self.text_processor.save(path_to_safe_dir)


    @classmethod
    def load_model(cls, path_to_safe_dir):
        processor_metadata_path = os.path.join(path_to_safe_dir, 'metadata.json')
        model_path = os.path.join(path_to_safe_dir, 'super_model.keras')
        processor = TextDataProcessor.load(processor_metadata_path)
        model = keras.models.load_model(model_path)
        return cls(text_processor=processor, model=model)
