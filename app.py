from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import json
from tensorflow import keras
import io

app = Flask(__name__)
CORS(app)

model_paths = {
    'model1': 'sentiment_model.h5',
    'model2': 'sentiment_model.h5',

    # Add more model paths as needed
}

# Initialize cache variables
last_loaded_model_name = None
last_loaded_model = None

class SentimentPredictor:
    def __init__(self, model_name, tokenizer_path):
        global last_loaded_model_name
        global last_loaded_model

        model_path = model_paths.get(model_name)
        if model_path is None:
            raise ValueError("Invalid model name")

        # Check if the requested model is already loaded
        if last_loaded_model_name != model_name:
            self.model = keras.models.load_model(model_path)
            last_loaded_model = self.model
            last_loaded_model_name = model_name
        else:
            self.model = last_loaded_model

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

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data['text']
    model_name = data['model']

    try:
        predictor = SentimentPredictor(model_name, 'tokenizer.json')
        sentiment = predictor.predict_sentiment(text)
        return jsonify({'sentiment': sentiment})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model_summary')
def get_model_summary():
    global last_loaded_model
    if last_loaded_model is None:
        return jsonify({'error': 'No model loaded'}), 400

    out = io.StringIO()
    last_loaded_model.summary(print_fn=lambda x: out.write(x + '\n'))
    summary_string = out.getvalue()
    return jsonify({'model_summary': summary_string})

MAX_SEQUENCE_LENGTH = 100

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, port=5001)