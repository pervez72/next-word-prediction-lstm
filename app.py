import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import os


# App Config
# ===============================
st.set_page_config(page_title="Next Word Predictor")


# Load Tokenizer safely
# ===============================
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        tokenizer_data = f.read()
    return tokenizer_from_json(tokenizer_data)

tokenizer = load_tokenizer()
vocab_size = len(tokenizer.word_index) + 1
max_sequence_len = 14  # Matches training


# Define and Load LSTM Model (Architecture recreated since JSON is missing)
# ===============================
@st.cache_resource
def load_model():
    weights_path = "model.weights.h5"

    if os.path.exists(weights_path):
        # Recreate the model architecture (common setup for next-word prediction LSTM)
        # Adjust embedding_dim, lstm_units if your training used different values
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sequence_len-1))
        model.add(LSTM(150, return_sequences=True))  # First LSTM layer
        model.add(LSTM(100))  # Second LSTM layer (optional; remove if your model had only one)
        model.add(Dense(vocab_size, activation='softmax'))

        # Load the saved weights
        model.load_weights(weights_path)

        # Compile the model
        model.compile(
            loss="categorical_crossentropy",
            optimizer='adam',
            metrics=['accuracy']
        )
        return model
    else:
        st.error(f"Weights file ({weights_path}) not found! Please upload your trained model weights.")
        return None

model = load_model()


# Next Word Prediction Function 
# =============================================
def predict_next_word(model, tokenizer, text, max_sequence_len):
    if model is None:  # Check if model was loaded successfully
        return "Error: Model not loaded."

    token_list = tokenizer.texts_to_sequences([text])[0]

    if not token_list:
        return None

    # Ensure sequence length matches training input
    if len(token_list) >= max_sequence_len - 1:
        token_list = token_list[-(max_sequence_len-1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted, axis=1)[0]

    # Reverse lookup from index → word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None 


# Streamlit UI
# ===============================
st.title("Next Word Prediction")
st.write("Type a sentence below and let the model predict the **next word**!")

input_text = st.text_input("Enter a sentence:", placeholder="Example: I love machine")

if st.button("Predict Next Word"):
    if input_text.strip():
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        if next_word:
            st.success(f"**Predicted Next Wod:** `{next_word}`")
        else:
            st.warning("⚠️ Sorry, could not predict a valid next word.")
    else:
        st.info("Please enter a valid text.")