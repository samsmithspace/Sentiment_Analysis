import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Assuming MAX_SEQUENCE_LENGTH is defined elsewhere or you need to define it
MAX_SEQUENCE_LENGTH = 100  # Example value, adjust according to your model's requirement


class SentimentPredictor:
    def __init__(self, model_path, tokenizer_path):
        self.model = keras.models.load_model(model_path)
        with open(tokenizer_path, 'r') as f:
            tokenizer_json = json.load(f)
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

    def preprocess_sentence(self, sentence):
        tokens = self.tokenizer.texts_to_sequences([sentence])
        return keras.preprocessing.sequence.pad_sequences(tokens, maxlen=MAX_SEQUENCE_LENGTH)

    def predict_sentiment(self, sentence):
        preprocessed = self.preprocess_sentence(sentence)
        prediction = self.model.predict(preprocessed)[0][0]
        if prediction >= 0.5:
            return "Positive"
        else:
            return "Negative"



if __name__ == "__main__":
    MODEL_PATH = 'sentiment_model.h5'
    TOKENIZER_PATH = 'tokenizer.json'

    predictor = SentimentPredictor(MODEL_PATH, TOKENIZER_PATH)

    while True:
        sentence = input("Enter a sentence (or type 'quit' to exit): ")
        if sentence.lower() == 'quit':
            break

        sentiment = predictor.predict_sentiment(sentence)
        print("Predicted sentiment:", sentiment)