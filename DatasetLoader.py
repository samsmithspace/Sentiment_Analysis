import json
import re
import nltk
import pandas as pd
import tensorflow as tf
from pandas.io import pickle
import matplotlib.pyplot as plt

nltk.download('punkt')  # For tokenization


class DatasetLoader:
    def __init__(self, tokenizer=None, tokenizer_path=None):
        if tokenizer_path:
            self.load_tokenizer(tokenizer_path)
        else:
            self.tokenizer = tokenizer

    def load_data(self, data_path, label_column='sentiment'):

        data = pd.read_csv(data_path)
        texts = data['reviewText'].tolist()
        labels = data[label_column].tolist()
        return texts, labels

    def preprocess_text(self, text):

        text = re.sub(r'\d+', '<NUM>', str(text))  # Replace numbers with placeholder
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = nltk.word_tokenize(text)

        # Optional: remove stopwords
        # stopwords = nltk.corpus.stopwords.words('english')
        # tokens = [w for w in tokens if w not in stopwords]
        return tokens

    def split_data(self, data, train_split=0.8):
        """Splits data into training and validation sets."""
        # ... (shuffle data before splitting)
        split_index = int(train_split * len(data))
        return data[:split_index], data[split_index:]

    def save_tokenizer(self, tokenizer_path='tokenizer.json'):
        """Saves the tokenizer to a JSON file."""
        with open(tokenizer_path, 'w') as f:
            json.dump(self.tokenizer.to_json(), f)

    def load_tokenizer(self, tokenizer_path='tokenizer.json'):
        """Loads a tokenizer from a JSON file."""
        with open(tokenizer_path, 'r') as f:
            tokenizer_json = json.load(f)
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

    def create_tokenizer(self, texts, max_words=10000):
        """Creates a tokenizer, fits it on the provided texts, and prints vocabulary size."""
        # Instantiate and fit the tokenizer
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
        self.tokenizer.fit_on_texts(texts)

        # Calculate vocabulary size
        vocab_size = len(self.tokenizer.word_index) + 1  # +1 for OOV token

        # Save the tokenizer after creation
        self.save_tokenizer()

        # Print and return the tokenizer's vocabulary size
        print("Vocabulary size:", vocab_size)
        return self.tokenizer

    def texts_to_sequences(self, texts):
        """Converts texts to sequences of token indices."""
        if not self.tokenizer:
            self.tokenizer = self.create_tokenizer(texts)
        return self.tokenizer.texts_to_sequences(texts)

    def pad_sequences(self, sequences, maxlen=100):
        """Pads sequences to a fixed length."""
        return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)

    def plot_word_frequencies(self, texts, top_n=10):
        """Plots the frequencies of the top 'top_n' most frequent words. """
        all_words = [word for text in texts for word in self.preprocess_text(text)]
        word_counts = nltk.FreqDist(all_words)
        most_common = word_counts.most_common(top_n)

        words, counts = zip(*most_common)
        plt.figure(figsize=(10, 5))  # Adjust figsize for readability
        plt.bar(words, counts)
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for clarity
        plt.ylabel("Frequency")
        plt.title("Top {} Most Frequent Words".format(top_n))
        plt.show()

    def process_and_pad_data(self, texts):

        processed_texts = [self.preprocess_text(text) for text in texts]
        sequences = self.texts_to_sequences(processed_texts)
        padded_sequences = self.pad_sequences(sequences)
        return padded_sequences

    def print_label_distribution(self, labels):
        """Prints the distribution of samples labeled with 1 and 0."""
        num_ones = sum(labels)
        num_zeros = len(labels) - num_ones

        print("Distribution of labels:")
        print(" - Number of samples labeled with 1:", num_ones)
        print(" - Number of samples labeled with 0:", num_zeros)


def load_tokenizer(vocab_path):
    """Loads a tokenizer from a saved file."""
    with open(vocab_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer
