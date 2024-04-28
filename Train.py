import os
# Allow interact with the operating system, like reading file paths.
import pandas as pd
# A library for data manipulation and analysis.
import numpy as np
# A library for numerical operations, useful for handling arrays.
import tensorflow as tf
# A library for machine learning and neural networks.

# These lines import specific tools from tensorflow to build and train neural networks.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
# A library for creating graphs and plots.
from sklearn.model_selection import train_test_split
# A tool from the scikit-learn library to split data into training and testing sets.
from typing import Tuple
# Allows you to specify that a function will return a tuple, which is a kind of list that can't be changed.
import logging
# A library for tracking events that happen when the program runs.

# This sets up how to record events, like errors or other important actions.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration settings for model and processing
CONFIG = {
    'max_features': 210000,         # Maximum number of unique words
    'sequence_length': 2000,        # Number of words in each input sequence
    'batch_size': 8,               # Number of training samples per batch to process
    'training_split': 0.7,          # Fraction of data to use for training
    'oversample_threshold': 1000    # Minimum number of samples per class for balancing
}

def load_text_data(filepath: str) -> pd.DataFrame:
    # If the file at 'filepath' doesn't exist, log an error and stop the program.
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"No such file: {filepath}")
    try:
        # Try reading the file into a DataFrame which can be used for data processing.
        return pd.read_csv(filepath)
    
    # If the file is empty, log an error and stop the program.
    except pd.errors.EmptyDataError:
        logging.error("File is empty: {}".format(filepath))
        raise
    # If the file is in the wrong format or corrupted, log an error and stop the program.
    except pd.errors.ParserError:
        logging.error("File is corrupt or unreadable: {}".format(filepath))
        raise

def preprocess_text_features(data_frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    features = data_frame['comment_text'].values  # Get all the text comments from the DataFrame
    labels = data_frame.iloc[:, 2:].values  # Get the categories starting from the third column onwards
    return features, labels  # Return the comments and their categories

def configure_vectorization_layer(text_samples: np.ndarray) -> TextVectorization:
    vectorizer = TextVectorization(
        max_tokens=CONFIG['max_features'],  # The most words (or tokens) it should learn about
        output_sequence_length=CONFIG['sequence_length'],  # How long each list of numbers should be
        output_mode='int'  # Turn text into integers (whole numbers)
    )
    vectorizer.adapt(text_samples) 
    return vectorizer

def augment_dataset_by_class(data_frame: pd.DataFrame) -> pd.DataFrame:
    for label in data_frame.columns[2:]:  # Look at each type of comment category from the third column onward
        minority_class_size = np.sum(data_frame[label] == 1)  # Count how many times this type appears
        oversample_needed = CONFIG['oversample_threshold'] - minority_class_size  # Determine how many more are needed
        if oversample_needed > 0:  # If more examples are needed,
            minority_data = data_frame[data_frame[label] == 1]  # Get the rows that have this type
            additional_samples = minority_data.sample(min(oversample_needed, len(minority_data)), replace=False, random_state=42)  # Randomly pick the needed amount
            data_frame = pd.concat([data_frame, additional_samples])  # Add these picked rows back to the main data
    return data_frame.sample(frac=1, random_state=42)  # Shuffle all data to mix the newly added ones evenly

def split_data_into_sets(vectorized_text: np.ndarray, labels: np.ndarray) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Use sklearn to split the data more robustly."""
    # Ensure data is in numpy format for sklearn operations
    if isinstance(vectorized_text, tf.Tensor):
        vectorized_text = vectorized_text.numpy()
    if isinstance(labels, tf.Tensor):
        labels = labels.numpy()

    x_train, x_val, y_train, y_val = train_test_split(vectorized_text, labels, test_size=1-CONFIG['training_split'], random_state=42)
    
    # Creating TensorFlow datasets from numpy arrays
    training_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).cache().shuffle(1000).batch(CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)
    validation_set = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)
    return training_set, validation_set

def build_and_compile_model(vocab_size) -> Model:
    """Create a brain-like network to understand and classify the text based on its content."""
    inputs = Input(shape=(None,), dtype='int32')  # Start by defining the input shape
    x = Embedding(input_dim=vocab_size, output_dim=128)(inputs)
    x = Bidirectional(LSTM(64, return_sequences=False, activation='tanh'))(x) # Apply a bidirectional LSTM layer that processes data in both forward and backward directions to maintain context.
    x = Dense(128, activation='relu')(x)  # Add layers that introduce non-linearity and ability to learn complex patterns
    x = Dense(256, activation='relu')(x)  # Another layer to increase the network's complexity and capacity
    x = Dense(128, activation='relu')(x)  # Further improve the features learned by the network
    outputs = Dense(6, activation='sigmoid')(x)  # Determine the output using sigmoid to classify into one of six categories
    model = Model(inputs=inputs, outputs=outputs)  # combine the model from input to output
    model.compile(
        loss='binary_crossentropy',  # Use binary crossentropy loss to measure how far off predictions are from actual labels
        optimizer='adam',  # Use the Adam optimizer for adjusting weights
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]  # Track accuracy, precision, and recall during training
    )
    return model

def train_model(model: Sequential, training_data: tf.data.Dataset, validation_data: tf.data.Dataset, save_path: str) -> dict:
    """
    Train the neural network and keep the best version of it.
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3),  # Stop if no improvement after 3 attempts
        ModelCheckpoint(save_path, save_best_only=True)  # Save only the best model to the file
    ]
    # Train the model using the training data, validate it using the validation data, and apply the callbacks
    history = model.fit(training_data, epochs=2, validation_data=validation_data, callbacks=callbacks)
    return history.history  # Return the performance records of the training process

def visualize_training_results(training_results: dict) -> None:
    """
    Show how the model's training went, by showing a graph 
    """
    plt.figure(figsize=(10, 6))  # Set the size of the plot
    for key, values in training_results.items():
        plt.plot(values, label=f'{key.capitalize()}')  # Plot each metric in the history
    plt.title('Model Training Metrics')  # Title of the plot
    plt.xlabel('Epochs')  # Label for the x-axis
    plt.ylabel('Value')  # Label for the y-axis
    plt.legend()  # Add a legend to clarify what each line represents
    plt.grid(True)  # Turn on the grid for easier reading of the plot
    plt.show()  # Display the plot

def execute():
    """ This main function handles loading data, preparing it, training the model, and showing results. """
    # Find the directory where this script is located.
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Set up the path to the training data CSV in the 'Toxic' folder.
    dataset_path = os.path.join(current_directory, 'Toxic', 'train.csv')
    # Set up the path to save the trained model.
    model_checkpoint_path = os.path.join(current_directory, 'model_checkpoint.h5')

    logging.info("Loading and preparing data...")
    # Load the data from the dataset path.
    dataframe = load_text_data(dataset_path)
    # Get text and labels from the data.
    features, labels = preprocess_text_features(dataframe)

    logging.info("Setting up text processing...")
    # Create a text processing layer.
    text_vectorizer = configure_vectorization_layer(features)
    # Process the text to make it ready for the model.
    vectorized_text = text_vectorizer(features)

    logging.info("Making classes equal...")
    # Make sure each class has the same amount of data.
    balanced_dataframe = augment_dataset_by_class(dataframe)
    # Get text and labels again from the balanced data.
    features, labels = preprocess_text_features(balanced_dataframe)
    # Process the balanced text.
    vectorized_text = text_vectorizer(features)

    logging.info("Splitting data...")
    # Split data into training and validation sets.
    train_set, validation_set = split_data_into_sets(vectorized_text, labels)

    logging.info("Setting up the model...")
    # Get the size of the vocabulary for the model.
    vocab_size = len(text_vectorizer.get_vocabulary())
    # Create and set up the model.
    neural_network = build_and_compile_model(vocab_size)

    logging.info("Training the model...")
    # Train the model with the training set and validate with the validation set.
    training_results = train_model(neural_network, train_set, validation_set, model_checkpoint_path)

    logging.info("Showing training results...")
    # Show how the training went.
    visualize_training_results(training_results)
    # Confirm the model is saved.
    logging.info(f"Model saved at {model_checkpoint_path}")

if __name__ == '__main__':
    execute()