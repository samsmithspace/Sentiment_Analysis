from sklearn.model_selection import StratifiedKFold

from Config import *
from Model import *
from DatasetLoader import *
def train_and_evaluate_model():
    loader = DatasetLoader()
    train_data, train_labels = loader.load_data(TRAIN_DATA_PATH)
    val_data, val_labels = loader.load_data(VAL_DATA_PATH)
    test_data, test_labels = loader.load_data(TEST_DATA_PATH)

    train_sequences = loader.process_and_pad_data(train_data)
    val_sequences = loader.process_and_pad_data(val_data)
    test_sequences = loader.process_and_pad_data(test_data)

    model = SentimentClassifier(VOCAB_SIZE, EMBEDDING_DIM, 1, tokenizer=loader.tokenizer)
    model.build_model()

    # Ensure sequences and labels are NumPy arrays
    train_sequences = np.array(train_sequences)
    train_labels = np.array(train_labels)
    val_sequences = np.array(val_sequences)
    val_labels = np.array(val_labels)
    test_sequences = np.array(test_sequences)
    test_labels = np.array(test_labels)
     # Set up TensorBoard callback
    #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Cross-validation
    kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for train_index, val_index in kfold.split(train_sequences, train_labels):
        # Use train_index to index NumPy arrays
        model.train((train_sequences[train_index], train_labels[train_index]),
                    (val_sequences, val_labels), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)  # Pass validation data as a tuple


    # Final evaluation
    #test_acc, test_f1 = model.evaluate(test_sequences, test_labels)
    #print("Test Accuracy:", test_acc)
    #print("Test F1:", test_f1)

    model.model.save(MODEL_SAVE_PATH)
if __name__ == "__main__":
    train_and_evaluate_model()