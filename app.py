import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import re  
import gradio as gr

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Zñḳḍāī\s]", "", text)
    text = re.sub(r'(\n)(\S)', r'\1 \2', text)
    return text

# Load the dataset
df = pd.read_csv('Roman-Urdu-Poetry.csv')
df['Poetry'] = df['Poetry'].apply(clean_text)

# Tokenization
tokenizer = Tokenizer(num_words=5000, filters='')
tokenizer.fit_on_texts(df['Poetry'])
sequences = tokenizer.texts_to_sequences(df['Poetry'])

max_sequence_len = max([len(seq) for seq in sequences])  
max_sequence_len = min(max_sequence_len, 225) 
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')
K.clear_session()

input_sequences = []
output_words = []

for seq in padded_sequences:
    for i in range(1, len(seq)):
        input_sequences.append(seq[:i])
        output_words.append(seq[i])

input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
output_words = np.array(output_words)
total_words = len(tokenizer.word_index) + 1


# Load the trained model
model = load_model('poetry_model.h5')

# Function to generate poetry
def generate_poem(seed_text, next_words, temperature):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predictions = model.predict(token_list, verbose=0)[0]
        
        # Apply temperature scaling
        predictions = np.log(predictions + 1e-10) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        # Sample the next word
        predicted_word_index = np.random.choice(len(predictions), p=predictions)
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')

        if predicted_word:
            seed_text += " " + predicted_word  

    return seed_text

# Custom CSS Styling
custom_css = """
body {
    background-color: #121212;
    color: white;
    font-family: 'Arial', sans-serif;
}
.gradio-container {
    max-width: 600px;
    margin: auto;
    text-align: center;
}
textarea, input, button {
    font-size: 16px !important;
}
button {
    background: #ff5c5c !important;
    color: white !important;
    padding: 12px 18px !important;
    border-radius: 8px !important;
    font-weight: bold;
    border: none !important;
    cursor: pointer;
}
button:hover {
    background: #e74c3c !important;
}
"""

# Gradio Interface
with gr.Blocks(css=custom_css) as iface:
    gr.Markdown("<h1 style='text-align: center;'>🎶 Verse Hub: Poetry Generator 🎶</h1>")
    

    seed_text = gr.Textbox(label="Verse Hub", placeholder="Start your poetry...", interactive=True)

    with gr.Row():
        words = gr.Slider(minimum=5, maximum=100, step=1, value=10, label="Number of Words", interactive=True)
        temperature = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Temperature", interactive=True)

    generate_button = gr.Button("✨ Generate Poetry 🎤")
    output_text = gr.Textbox(label="Generated Poem", interactive=False, lines=6)

    generate_button.click(fn=generate_poem, inputs=[seed_text, words, temperature], outputs=output_text)

# Launch Gradio App
iface.launch()
