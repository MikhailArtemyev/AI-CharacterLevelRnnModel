# **Bidirectional GRU Name Generator**
**Note:** This is the second, more advanced version of this project.  (September 2025)

A TensorFlow/Keras implementation of a character-level bidirectional GRU neural network for generating human-like names using deep learning.

## **Overview**

This project implements a character-level recurrent neural network that learns to generate realistic human names by learning patterns from a dataset of existing names. 
The model uses bidirectional GRUs to capture both forward and backward dependencies in character sequences, making it particularly effective for name generation.

## **Architecture Choices**

### **Model Architecture**

**Bidirectional GRU Layers**: The model uses stacked bidirectional GRU layers (default: [128, 64] units) that process sequences in both forward and backward directions. 

**Character Embedding**: Characters are mapped to dense 32-dimensional vector representations, allowing the model to learn relationships between characters more efficiently than one-hot encoding.

**Batch Normalization**: Applied after each GRU layer to stabilize training and accelerate convergence, particularly important when training on diverse name datasets.

**Dense Classification Layers**: Two fully-connected layers (128 and 64 units) with ReLU activation process the GRU outputs before the final softmax classification.

**Dropout Regularization**: 0.3 dropout rate throughout the network prevents overfitting and improves generalization to generate novel names.

### **Training Strategy**

**Sequence-to-Sequence Learning**: For each position in a name, the model learns to predict the next character given all previous characters, creating a autoregressive generation capability.

**Special Tokens**: 
- `^` (start token): Marks the beginning of a name
- `$` (end token): Marks the end of a name
- Space padding: Ensures consistent sequence lengths

**Adaptive Learning Rate**: Uses ReduceLROnPlateau callback to automatically adjust learning rate when validation loss plateaus, with early stopping to prevent overfitting.

**Data Requirements**: Works best with 1000+ unique names. The model can handle diverse cultural naming patterns and learns length distributions automatically.

### **Generation Mechanism**

**Temperature-Controlled Sampling**: 
- **Low temperature (0.3-0.5)**: Conservative, common name patterns
- **Medium temperature (0.8-1.0)**: Balanced creativity and realism
- **High temperature (1.2-1.5)**: More creative and unusual names

**Autoregressive Generation**: Names are generated character-by-character, with each prediction conditioned on all previous characters until the end token is generated or maximum length is reached.

**Starting Letter Support**: Can generate names beginning with specific letters or letter combinations.

## **Files**

* `main.py`: Program entry point
* `BiDir_SuperModel.py.py`: Complete implementation of the model with SuperRNN class
* `InputProcessor.py`: Contains a helper class for working with text input  
* `names.txt`: Training dataset (one name per line)
* `super_model/`: Directory containing saved model and metadata
  - `model.keras`: Trained Keras model
  - `metadata.json`: Character mappings and configuration

## **Usage**

### **Training a New Model**

```python
from BiDir_SuperModel import SuperRNN
from InputProcessor import TextDataProcessor

# Max length of a word
max_length = 15 

# Initialize generator
processor = TextDataProcessor("names.txt", max_length)
super_model = SuperRNN(
    text_processor=processor,
    gru_units=[128, 64],
    dropout_rate=0.3,
    learning_rate=0.001
)

# Train the model
history = super_model.train(
    epochs=100,
    batch_size=128,
    validation_split=0.2
)

# Save the trained model
super_model.save_model('super_model')
```

### **Loading and Generating Names**

```python
from BiDir_SuperModel import SuperRNN

# Load model
super_model = SuperRNN.load_model("super_model_dir")

# Generate single name
name = super_model.generate(temperature=1.0, start_char='M')
print(f"Generated: {name}")

```


## **Key Features**

* **Bidirectional Processing**: Captures both forward and backward character dependencies
* **Cultural Diversity**: Learns patterns from names across different cultures
* **Variable Length Generation**: Automatically learns appropriate name lengths
* **Temperature Control**: Fine-tune creativity vs. realism in generated names
* **Batch Generation**: Efficiently generate multiple names at once
* **Model Versioning**: Save and load multiple model versions with metadata
* **Starting Letter Constraints**: Generate names beginning with specific characters

## **Model Parameters**

* `max_length`: 15 (maximum name length)
* `embedding_dim`: 32 (character embedding size)
* `gru_units`: [128, 64] (hidden units per layer)
* `dropout_rate`: 0.3
* `learning_rate`: 0.001
* `batch_size`: 128
* `early_stopping_patience`: 10-20 epochs

## **Performance Metrics**

With a dataset of 2000+ names, expect:
* Training loss: 1.5-2.0
* Validation accuracy: 35-45%
* Generation quality: Realistic, pronounceable names

## **Requirements**

* TensorFlow 2.x
* Keras
* NumPy
* Python 3.7+

## **Optional Requirements**

* matplotlib (for training visualization)

## **Dataset Recommendations**

For best results, your `names.txt` should contain:
* At least 1000 unique names (2000+ recommended)
* One name per line
* Consistent formatting (e.g., all lowercase or properly capitalized)

## **License**

This project is for educational purposes. Model trained on publicly available name datasets.

## **Acknowledgments**

- Inspired by Andrej Karpathy's char-rnn work
- Architecture influenced by sequence-to-sequence modeling techniques
- Training strategies adapted from NLP best practices