import json
import pandas as pd
import random

class JSONToCSVTransformer:
    def __init__(self, json_path, train_split=0.8, val_split=0.1, test_split=0.1):

        self.json_path = json_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def process_review(self, review):

        text = review.get('reviewText', "")  # Use .get() to provide a default value
        overall_rating = review['overall']
        sentiment = 1 if overall_rating > 2.5 else 0
        return pd.Series({'reviewText': text, 'sentiment': sentiment})

    def transform_and_save(self, output_dir):
        """
        Loads the JSON data, transforms it, splits into datasets, and saves as CSV files.

        Args:
          output_dir (str): The directory to save the output CSV files.
        """
        data = []
        with open(self.json_path, 'r') as f:
            for line in f:
                review = json.loads(line)
                data.append(self.process_review(review))

        df = pd.DataFrame(data)

        # Shuffle before splitting (optional, but good practice)
        df = df.sample(frac=1).reset_index(drop=True)

        # Calculate split indices
        data_len = len(df)
        train_end = int(self.train_split * data_len)
        val_end = train_end + int(self.val_split * data_len)

        # Create datasets
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        # Save as CSV files
        train_df.to_csv(output_dir + '/train.csv', index=False)
        val_df.to_csv(output_dir + '/val.csv', index=False)
        test_df.to_csv(output_dir + '/test.csv', index=False)

if __name__ == "__main__":
    transformer = JSONToCSVTransformer("AMAZON_FASHION.json")
    transformer.transform_and_save(output_dir=".")  # Save to the current directory