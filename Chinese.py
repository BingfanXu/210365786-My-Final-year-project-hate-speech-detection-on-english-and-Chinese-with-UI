import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
import jieba

# This is for configuration
PARAMS = {
    'features': 210000,
    'length': 2000,
    'batch_size': 4
}

def segment_texts(data, column):
    # This function uses jieba, a library for segmenting Chinese text. This is to split text into individual words or phrases.
    return data[column].apply(lambda text: ' '.join(jieba.cut(text)))

def vectorize_texts(texts, config):
    # This function sets up a TextVectorization layer in TensorFlow to convert text into numerical data that a model can process.
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=config['features'],  # Sets the maximum number of unique words.
        output_sequence_length=config['length']  # Sets the length of the resulting numerical vectors, which is 2000 in this case
    )
    vectorizer.adapt(texts)  # This learns the vocabulary from the texts and prepares the vectorizer.
    return vectorizer(texts), vectorizer  # Returns both the vectorized texts and the vectorizer itself, so it can be used in interface.

def assemble_datasets(text_vectors, labels):
    # This function creates a TensorFlow dataset from the provided text vectors and labels, which is useful for training or evaluation.
    return tf.data.Dataset.from_tensor_slices((text_vectors, labels)).batch(PARAMS['batch_size'])  # Organizes data into batches of a specified size.

def define_model(vocab_size):
    # Start defining a machine learning model that will input text data.
    inputs = Input(shape=(None,), dtype=tf.int32)
    # Embed the input sequence into a space that our model can work with more easily.
    x = Embedding(input_dim=vocab_size, output_dim=128)(inputs)
    # Apply a bidirectional LSTM layer that processes data in both forward and backward directions to maintain context.
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    # Add another bidirectional LSTM layer to further capture dependencies in data not captured by the first layer.
    x = Bidirectional(LSTM(64))(x)
    # Introduce a connected layer that can help in classification by learning non-linear combinations of features.
    x = Dense(64, activation='relu')(x)
    # Define the output layer that uses sigmoid activation to predict if the input text is in one class or another, such as toxic or non-toxic.
    outputs = Dense(1, activation='sigmoid')(x)
    # combine the layers into a model and specify how it should learn using Adam optimization and binary crossentropy loss.
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
    # Return the fully defined model ready for training.
    return model

def save_trained_model(model, path='model_saved.h5'):
    # Save the final model so I can compare performance between best and current one.
    model.save(path)
    # Print a message shows where the model was saved.
    print(f"Model saved to {path}")

def plot_results(history):
    """Plots training and validation metrics."""
    plt.figure(figsize=(12, 5))
    for key, value in history.items():
        plt.plot(value, label=key)
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)
    plt.show()

def process_and_train():
    #read the dataset
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_data = pd.read_csv(os.path.join(base_path, 'ChineseToxic', 'train.csv'))
    validation_data = pd.read_csv(os.path.join(base_path, 'ChineseToxic', 'dev.csv'))

    # Process text data
    train_data['processed'] = segment_texts(train_data, 'TEXT')
    validation_data['processed'] = segment_texts(validation_data, 'TEXT')

    # Vectorize text data
    train_vectors, vectorizer = vectorize_texts(train_data['processed'].values, PARAMS)
    validation_vectors = vectorizer(validation_data['processed'].values)

    # Create datasets
    train_dataset = assemble_datasets(train_vectors, train_data['label'].values)
    validation_dataset = assemble_datasets(validation_vectors, validation_data['label'].values)

    # Build and train model
    model = define_model(len(vectorizer.get_vocabulary()))
    model.summary()
    #early stop and save the model on each epoch if performance is better
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3),
        ModelCheckpoint('Chinese_model.h5', save_best_only=True)
    ]
    #train and plot then save the last model independently.
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=1, callbacks=callbacks)
    plot_results(history.history)
    save_trained_model(model)

if __name__ == '__main__':
    process_and_train()