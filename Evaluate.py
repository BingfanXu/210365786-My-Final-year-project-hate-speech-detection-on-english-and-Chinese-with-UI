import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.layers import TextVectorization

# Define the evaluation function
def evaluate_model_per_label(model, vectorizer, test_data_path, threshold):
    # Load test data
    test_data = pd.read_csv(test_data_path)
    test_features = test_data['comment_text'].values
    # Convert the labels needed to binary based on a threshold
    test_labels = (test_data.iloc[:, 2:] > 0.5).astype(int)

    # Preprocess test data
    test_features_vectorized = vectorizer(test_features)

    # Predict on the test data
    predictions = model.predict(test_features_vectorized)

    # Convert predictions to binary values using the provided threshold
    binary_predictions = (predictions > threshold).astype(int)

    # Calculate and print the evaluation metrics for each label
    metrics_per_label = {}
    for label in test_labels.columns:
        label_idx = test_labels.columns.get_loc(label)
        label_truth = test_labels[label].values
        label_predictions = binary_predictions[:, label_idx]
        accuracy = accuracy_score(label_truth, label_predictions)
        f1 = f1_score(label_truth, label_predictions, zero_division=0)
        metrics_per_label[label] = {'Accuracy': accuracy, 'F1-Score': f1}
        print(f'Label: {label} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}')

    # Calculate and print summary metrics for all labels, because I don't want to calculate summary result by myself.
    summary_accuracy = accuracy_score(test_labels, binary_predictions)
    summary_precision = precision_score(test_labels, binary_predictions, average='macro', zero_division=0)
    summary_recall = recall_score(test_labels, binary_predictions, average='macro')
    summary_f1 = f1_score(test_labels, binary_predictions, average='macro', zero_division=0)

    print(f'\nSummary Metrics - Accuracy: {summary_accuracy:.4f}, Precision: {summary_precision:.4f}, Recall: {summary_recall:.4f}, F1-Score: {summary_f1:.4f}')

    return metrics_per_label, {'Summary Accuracy': summary_accuracy, 'Summary Precision': summary_precision, 'Summary Recall': summary_recall, 'Summary F1-Score': summary_f1}

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    english_model_path = os.path.join(script_dir, 'model_checkpoint.h5')
    english_test_data_path = os.path.join(script_dir, 'Toxic', 'merged_test_submission.csv')  # Path to English test data

    # Load models
    english_model = tf.keras.models.load_model(english_model_path)

    # English Text Vectorization Setup, explained in interface page already
    MAX_FEATURES = 200000 
    MAX_SEQUENCE_LENGTH = 1800  
    english_vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=MAX_SEQUENCE_LENGTH, output_mode='int')

    # Adapt the English TextVectorization layer
    df_english = pd.read_csv(english_test_data_path)
    english_text_data = df_english['comment_text']
    english_vectorizer.adapt(english_text_data.values)
    # Set the threshold value
    threshold = 0.5

    # Evaluate the English model for each label
    print("Evaluating English Model for each label with threshold:", threshold)
    evaluate_model_per_label(english_model, english_vectorizer, english_test_data_path, threshold)