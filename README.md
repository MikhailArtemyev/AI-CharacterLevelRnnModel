# Character-Level RNN Text Generator

A TensorFlow/Keras implementation of a character-level recurrent neural network for text generation, designed for generating names or short text sequences.

## Overview

This project implements a character-level RNN that learns to generate text one character at a time. It's particularly suited for generating names, words, or other short text sequences by learning patterns from training data.

## Architecture Choices

### Model Architecture

**Dual-Layer LSTM Design**: The model uses two stacked LSTM layers (configurable to GRU or SimpleRNN) for capturing both short-term and long-term dependencies in character sequences. The dual-layer approach allows the network to learn more complex patterns than a single layer would permit.

**Embedding Layer**: Characters are first mapped to dense vector representations (64-dimensional by default) rather than using one-hot encoding. This reduces dimensionality and allows the model to learn semantic relationships between characters.

**Dropout Regularization**: A 0.2 dropout rate prevents overfitting by randomly dropping connections during training, improving generalization to unseen sequences.

**Dense Output Layer**: The final layer outputs probabilities over the entire vocabulary, enabling character-by-character generation.

### Training Strategy

**Sequential Processing**: The model processes examples one at a time with immediate gradient updates, similar to online learning. This approach works well for small datasets like name lists.

**Dynamic Padding**: Input sequences are padded to a consistent length (default 10 characters) to enable batch processing while maintaining variable-length generation capability.

**Custom Loss Function**: Uses sparse categorical crossentropy with masking support to handle variable-length sequences efficiently.

### Generation Mechanism

**Temperature-Controlled Sampling**: The temperature parameter (0.1-2.0) controls randomness:
- Low temperature (0.1): Conservative, high-confidence predictions
- High temperature (2.0): More creative, diverse outputs

**Stop Character**: A designated stop character (`,` by default) allows the model to learn sequence boundaries and generate variable-length outputs naturally.

## Files

- `super_model.py`: Core model implementation with LSTM/GRU architectures
- `train.py`: Data processing, training loop, and model persistence
- `names.txt`: Training data (expected format: comma-separated names)

## Usage

### Training a New Model

```python
from train import create_model, save_complete_model

# Load and preprocess data
text = open('names.txt', 'r').read().lower().replace(' ', '')
model, processor = create_model(text)

# Train the model
names = [w + ',' for w in text.split(',')]
model.train_model(processor, names, num_iterations=10000)

# Save everything
save_complete_model(model, processor, 'super_model')
```

### Loading and Using a Trained Model

```python
from train import load_complete_model

# Load saved model
model, processor = load_complete_model('super_model')

# Generate text
generated = model.generate_text(
    start_string="john",
    char_to_idx=processor.char_to_idx,
    idx_to_char=processor.idx_to_char,
    num_generate=5,
    temperature=0.8
)
```

## Key Features

- **Flexible RNN Types**: Support for LSTM, GRU, or SimpleRNN architectures
- **Stateful Generation**: Optional stateful mode for maintaining context across batches
- **Dual Persistence**: Models saved in both Keras format and separate weights/config files
- **Vocabulary Mapping**: Character mappings preserved in JSON and pickle formats for compatibility

## Requirements

- TensorFlow 2.x
- Keras
- NumPy

## Model Parameters

- `vocab_size`: Automatically determined from training data
- `embedding_dim`: 64 (character embedding size)
- `rnn_units`: 128 (hidden state size)
- `learning_rate`: 0.01
- `dropout`: 0.2
- `sequence_length`: 10 (for training padding)

## Licence 

This project is meant for educational purposes only.

## Acknowledgments
This work is highly inspired by DeepLearning.AI Specialization:
https://www.coursera.org/learn/nlp-sequence-models


