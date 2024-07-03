import asyncio
import csv
from string import printable
from tensorflow.keras.preprocessing import sequence
import numpy as np

PATH_TO_MISCLASSIFIED_URLS = 'C:/Users/larisabalc/Desktop/phishing_detector/datasets/misclassified_urls.csv'

MAX_LEN = 75
EPOCHS = 5
BATCH_SIZE = 32
TEST_SIZE = 0.25
RANDOM_STATE = 33

class RetrainUtils:
    """
    Utility class for retraining models and handling misclassified URLs.

    Methods:
        save_misclassified_urls(url, label, filename=PATH_TO_MISCLASSIFIED_URLS):
            Save misclassified URLs and their corresponding labels to a CSV file.

        combine_misclassified_urls_with_train_set(X_train, target_train, misclassified_urls):
            Combine misclassified URLs with the training set.

        train_with_misclassified_urls(model, X_train, target_train, X_test, target_test, misclassified_urls, websocket, epochs=EPOCHS, batch_size=BATCH_SIZE):
            Train the model with misclassified URLs added to the training set.
    """

    @staticmethod
    def save_misclassified_urls(url, label, filename=PATH_TO_MISCLASSIFIED_URLS):
        """
        Save misclassified URLs and their corresponding labels to a CSV file.

        Args:
            url (str): The URL that was misclassified.
            label (int): The label indicating whether the URL is benign (0) or malicious (1).
            filename (str, optional): Path to the CSV file to save misclassified URLs.
                                      Defaults to PATH_TO_MISCLASSIFIED_URLS.
        """
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            if csvfile.tell() == 0:
                csv_writer.writerow(['URLs', 'Label'])
            
            csv_writer.writerow([url, label])

    @staticmethod
    def combine_misclassified_urls_with_train_set(X_train, target_train, misclassified_urls):
        """
        Combine misclassified URLs with the training set.

        Args:
            X_train (numpy.ndarray): Features of the original training set.
            target_train (numpy.ndarray): Labels of the original training set.
            misclassified_urls (pandas.DataFrame): DataFrame containing misclassified URLs.

        Returns:
            numpy.ndarray, numpy.ndarray: Combined features and labels for the training set.
        """
        url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in misclassified_urls.URLs]

        X = sequence.pad_sequences(url_int_tokens, maxlen=MAX_LEN)

        target = np.array(misclassified_urls.Label)
        combined_X_train = np.concatenate([X_train, X])
        combined_target_train = np.concatenate([target_train, target])

        return combined_X_train, combined_target_train

    @staticmethod
    async def train_with_misclassified_urls(model, X_train, target_train, X_test, target_test, misclassified_urls, websocket, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """
        Train the model with misclassified URLs added to the training set.

        Args:
            model: The machine learning model to be trained.
            X_train (numpy.ndarray): Features of the original training set.
            target_train (numpy.ndarray): Labels of the original training set.
            X_test (numpy.ndarray): Features of the testing set.
            target_test (numpy.ndarray): Labels of the testing set.
            misclassified_urls (pandas.DataFrame): DataFrame containing misclassified URLs.
            websocket: WebSocket connection for sending training progress updates.
            epochs (int, optional): Number of epochs for training. Defaults to EPOCHS.
            batch_size (int, optional): Batch size for training. Defaults to BATCH_SIZE.
        """
        from webservice.websocket_handler import WebSocketHandler
        combined_X_train, combined_target_train = RetrainUtils.combine_misclassified_urls_with_train_set(X_train, target_train, misclassified_urls)
        
        num_batches = len(combined_X_train) // batch_size
        total_iterations = epochs * num_batches
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            iteration = 0
            
            for i in range(0, len(combined_X_train), batch_size):
                iteration += 1
                
                batch_X = combined_X_train[i:i+batch_size]
                batch_target = combined_target_train[i:i+batch_size]
                
                model.train_on_batch(batch_X, batch_target)
                
                progress_percentage = ((epoch * num_batches + iteration) / total_iterations) * 100
                
                progress_data = {
                    "type": 'TrainingProgress',
                    "progress": progress_percentage
                }

                print(progress_data)
                await WebSocketHandler.send_message(websocket, progress_data)
                await asyncio.sleep(0.5)
            
            loss, accuracy = model.evaluate(X_test, target_test, verbose=1)
            print('\nFinal Cross-Validation Accuracy:', accuracy, '\n')
            print('\nFinal Loss:', loss, '\n')