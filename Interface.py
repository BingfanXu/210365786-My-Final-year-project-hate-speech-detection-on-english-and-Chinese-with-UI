import os
import threading
import tensorflow as tf
import gradio as gr
import pandas as pd
from flask import Flask, render_template
from tensorflow.keras.layers import TextVectorization
import jieba


def tokenize_with_jieba(text):
    # Use the jieba library to split Chinese text into individual words or phrases.
    return " ".join(jieba.cut(text))

def score_comment(comment, language_index):
    # Check if the language of the comment is Chinese based on the language_index.
    if language_index == 1:
        # Tokenize the comment using the jieba tokenizer designed for Chinese text.
        tokenized_comment = tokenize_with_jieba(comment)
        # Convert the tokenized text into a numerical format the model can understand.
        vectorized_comment = chinese_vectorizer([tokenized_comment])
        # Use the trained Chinese model to predict the toxicity of the comment.
        prediction = chinese_model.predict(vectorized_comment)
        # Determine if the comment is toxic based on a predefined threshold.
        is_toxic = prediction[0][0] > 0.245
        # Convert the prediction result to a standard float type.
        confidence = float(prediction[0][0])
        # Organize the results into a dictionary indicating the toxicity and confidence levels. which can be present as graph
        result = {
            'Hate': confidence if is_toxic else 0,
            'Non-Hate': 1.0 - confidence if not is_toxic else 0.0
        }
    else:
        # If the comment is in English, vectorize the comment directly without tokenization.
        vectorized_comment = english_vectorizer([comment])
        # Use the trained English model to predict various types of toxicity in the comment.
        prediction = english_model.predict(vectorized_comment)
        # Organize the predictions into a dictionary with clear labels for each type of toxicity.
        result = {
            'Hate Speech': float(prediction[0][0]),
            'Several Label': float(prediction[0][1]),
            'Obscene': float(prediction[0][2]),
            'Threat': float(prediction[0][3]),
            'Insult': float(prediction[0][4]),
            'Identity Hate': float(prediction[0][5])
        }
    # Return the dictionary containing the toxicity scores.
    return result

def run_gradio_interface():
    #for some reason this CSS does not seems to work, but I leave it here to show my idea on my ideal interface.
    custom_css = """
    /* Custom CSS for Gradio Interface */
    /* Custom CSS for Gradio Interface */
    body, .gradio_app {  /* Apply the background to both the body and the Gradio app container */
        font-family: 'Arial', sans-serif;
        background-image: url('https://cdn.dribbble.com/users/1084794/screenshots/5443614/dimbud-drib-1.gif'); /* Add your image URL here */
        background-size: cover; /* Cover the entire area */
        background-repeat: no-repeat; /* Do not repeat the image */
        background-attachment: fixed; /* Fix the background relative to the viewport */
        background-position: center; /* Center the background image */
    }
    .gr-button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .gr-button:hover {
        background-color: #45a049;
    }
    """
    #this si the interface page for gradio, it turns the dictionary output percentage to graph in outputs=gr.components.Label(num_top_classes=3, type="confidences"),
    iface = gr.Interface(
        fn=score_comment,
        inputs=[
            gr.components.Textbox(lines=2, placeholder="Type a comment here...", show_label=False),
            gr.components.Radio(choices=["English", "Chinese"], label="Select Language", type="index")
        ],
        outputs=gr.components.Label(num_top_classes=3, type="confidences"),
        title="Hate speech detection Classifier/仇恨言论检测器",
        description="Select the Chinese language of the comment if you want detection otherwise will not work.\默认是英文，如果要中文记得选",
        css=custom_css
    )
    iface.launch()

#running flask that allows run HTML, CSS and Javascript as well
app = Flask(__name__)
#access for home.html
@app.route('/')
def home():
    return render_template('home.html')
#access for about.html
@app.route('/about')
def about():
    return render_template('about.html')
#access for contact.html
@app.route('/contact')
def contact():
    return render_template('contact.html')
#access for model.html, which is where Gradio is launched
@app.route('/model')
def model_page():
    # Embed Gradio web interface into Flask using iframe
    gradio_url = "http://127.0.0.1:7860"  # URL where Gradio interface is running
    return render_template('model.html', gradio_url=gradio_url)

if __name__ == "__main__":
    # Define directory paths and load the pre-trained models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    english_model_path = os.path.join(script_dir, 'model_checkpoint.h5')
    chinese_model_path = os.path.join(script_dir, 'Chinese_model.h5')

    english_model = tf.keras.models.load_model(english_model_path)
    chinese_model = tf.keras.models.load_model(chinese_model_path)

    # English Text Vectorization Setup
    MAX_FEATURES = 200000  # This is the maximum number of words in the vocabulary
    MAX_SEQUENCE_LENGTH = 1800  # This should be the length of the sequences after vectorization
    english_vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')
    english_csv_path = os.path.join(script_dir, 'Toxic', 'train.csv')
    df_english = pd.read_csv(english_csv_path)
    english_text_data = df_english['comment_text']
    english_vectorizer.adapt(english_text_data.values)

    # Initialize and adapt the TextVectorization layer
    chinese_vectorizer = TextVectorization(
        max_tokens=MAX_FEATURES,
        output_sequence_length=MAX_SEQUENCE_LENGTH,
        output_mode='int',
        standardize=None,
        split='whitespace'
    )
    # Assuming the existence of a training CSV file to adapt the vectorizer
    csv_path = os.path.join(script_dir, 'ChineseToxic', 'train.csv')
    df = pd.read_csv(csv_path)
    df['text_tokenized'] = df['TEXT'].apply(tokenize_with_jieba)  # Adjust 'TEXT' to your column name
    chinese_vectorizer.adapt(df['text_tokenized'].values)
    thread_gradio = threading.Thread(target=run_gradio_interface)
    thread_gradio.start()
    app.run(debug=True, use_reloader=False)  # use_reloader=False to avoid duplicate launches