import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.utils import class_weight


class SentimentClassifier:
    def __init__(self, vocab_size, embedding_dim, output_dim, model_type='lstm', tokenizer=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.model_type = model_type
        self.tokenizer = tokenizer  # Initialize tokenizer here
        self.model = None
        self.max_length = None  # To store sequence length after data loading

    def build_model(self):
        if self.model_type == 'lstm':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim),
                tf.keras.layers.LSTM(64),  # Example units in LSTM
                tf.keras.layers.Dense(1, activation='sigmoid')  # For binary sentiment
            ])
        elif self.model_type == 'cnn':
            # TODO: Implement a CNN architecture
            pass
        else:
            raise ValueError("Invalid model_type. Choose 'lstm' or 'cnn'")

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, train_data, validation_data, epochs=5, batch_size=32):
        # Updated to accommodate padded sequences directly

        X_train, y_train = train_data
        X_val, y_val = validation_data

        # Important: Store max_length for later prediction consistency
        self.max_length = X_train.shape[1]
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        self.model.fit(X_train, np.array(y_train), class_weight=class_weight_dict, epochs=epochs, batch_size=batch_size,
                       validation_data=(X_val, np.array(y_val)))

    def evaluate(self, test_data):
        # Assuming test_data is a tuple of (texts, labels)
        X_test, y_test = test_data
        return self.model.evaluate(X_test, np.array(y_test))

    def predict(self, text):
        # Preprocess the text with the same tokenizer used during training
        tokenized_text = self.tokenizer.texts_to_sequences([text])
        padded_text = pad_sequences(tokenized_text, maxlen=self.max_length)  # Assuming you've set max_length
        return self.model.predict(padded_text)[0][0]  # Get the probability/score