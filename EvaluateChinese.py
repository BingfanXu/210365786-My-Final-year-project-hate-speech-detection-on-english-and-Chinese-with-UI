import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.layers import TextVectorization


# Define the evaluation function
def evaluate_model_on_toxic(model, vectorizer, test_data_path, threshold):
    # Load test data
    toxic_data = pd.read_csv(test_data_path)
    print("Total entries in test dataset:", len(toxic_data))
    # Get the features and labels
    test_features = toxic_data['TEXT'].values
    test_labels = toxic_data['label'].values

    # Preprocess test data
    test_features_vectorized = vectorizer(test_features)

    # Predict on the test data
    predictions = model.predict(test_features_vectorized)

    # Convert predictions to binary values using the provided threshold
    binary_predictions = (predictions > threshold).astype(int)

    # Calculate and print the evaluation metrics
    accuracy = accuracy_score(test_labels, binary_predictions)
    f1 = f1_score(test_labels, binary_predictions, zero_division=0)

    print(f'Toxic Comments - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}')

    return {'Accuracy': accuracy, 'F1-Score': f1}

if __name__ == '__main__':
    # Define paths for your model and test data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    english_model_path = os.path.join(script_dir, 'Chinese_model.h5')
    english_test_data_path = os.path.join(script_dir, 'ChineseToxic', 'test.csv')

    # Load your model
    english_model = tf.keras.models.load_model(english_model_path)

    # English Text Vectorization Setup
    MAX_FEATURES = 200000  # This should match the training setup
    MAX_SEQUENCE_LENGTH = 1800  # This should match the training setup
    english_vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=MAX_SEQUENCE_LENGTH, output_mode='int')

    # Adapt the English TextVectorization layer
    df_english = pd.read_csv(english_test_data_path)
    english_text_data = df_english['TEXT']
    english_vectorizer.adapt(english_text_data.values)
    # Set the threshold value
    threshold = 0.245

    # Evaluate the model for toxic comments
    print("Evaluating English Model for toxic comments with threshold:", threshold)
    evaluate_model_on_toxic(english_model, english_vectorizer, english_test_data_path, threshold)