# Standard library imports
import asyncio
import csv
import json
import socket
from urllib.parse import urlparse

# Third-party library imports
import numpy as np
import pandas as pd
import requests
import torch
from ipwhois import IPWhois
from keras.callbacks import Callback
from sklearn import metrics, model_selection
from sklearn.model_selection import train_test_split
from string import printable
from tensorflow.keras.preprocessing import sequence
import websockets
import matplotlib.pyplot as plt
from transformers import BertTokenizer

# Phishing Email Detector imports
from phishing_email_detector.text_classification_dataset import TextClassificationDataset
from phishing_email_detector.bert_classifier import BERTClassifier
from phishing_email_detector.phishing_email_detection_utils import (
    evaluate,
    load_data,
    predict_is_phishing,
    train,
)

# Phishing URL Detector imports
from phishing_url_detector.phishing_url_detection_utils import (
    print_layers_dims,
    save_model,
    load_model,
)
from phishing_url_detector.lstm_classifier import LSTMClassifier
from phishing_url_detector.retrain_utils import RetrainUtils

# Cryptography imports
from webservice.crypto_utils import CryptoUtils

PATH_TO_URL_DATASET = 'C:/Users/larisabalc/Desktop/phishing_detector/datasets/phishing_and_benign_websites.csv'
PATH_TO_MISCLASSIFIED_URLS = 'C:/Users/larisabalc/Desktop/phishing_detector/datasets/misclassified_urls.csv'

EMAIL_MODEL_PATH = "phishing_email_detector/models/bert_classifier.pth"
EMAIL_MODEL_NAME = 'bert-base-uncased'
URL_MODEL_NAME = "LSTM"

MAX_LEN = 75
EPOCHS = 2
BATCH_SIZE = 256
MAX_LEN_MISCLASSIFIED_URLS = 1
TEST_SIZE = 0.25
RANDOM_STATE = 33

#Load models
tokenizer = BertTokenizer.from_pretrained(EMAIL_MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
email_classifier_model = BERTClassifier(EMAIL_MODEL_NAME, num_classes=2)
email_classifier_model.load_state_dict(torch.load(EMAIL_MODEL_PATH, map_location=device))
email_classifier_model.eval()
url_classifier_model = load_model('phishing_url_detector/models/' + URL_MODEL_NAME + ".json", 'phishing_url_detector/models/' + URL_MODEL_NAME + ".h5")

class WebSocketHandler:
    """
    Utility class for handling WebSocket connections and sending messages securely.

    Methods:
        send_message(websocket, json_data):
            Encrypts and sends a JSON message via WebSocket.

        handle_connection(websocket):
            Handles incoming messages over a WebSocket connection.
    """

    @staticmethod
    async def send_message(websocket, json_data):
        """
        Encrypts and sends a JSON message via WebSocket.

        Args:
            websocket: WebSocket connection.
            json_data (dict): JSON data to be sent.
        """
        encrypted_data = CryptoUtils.encrypt_message(json_data)
        await websocket.send(encrypted_data)

    @staticmethod
    async def handle_connection(websocket):
        """
        Handles incoming messages over a WebSocket connection.

        Args:
            websocket: WebSocket connection.
        """
        while True:
            encrypted_object = await websocket.recv()

            try:    
                decrypted_object = CryptoUtils.decrypt_message(encrypted_object)
                data = json.loads(decrypted_object)
            except Exception as e:
                print("Error decrypting:", e)
                continue

            if 'email' in data and data['type'] == 'CheckEmail':
                email_text = data['email']
        
                result = predict_is_phishing(email_text, email_classifier_model, tokenizer, device)

                json_data = {
                    "email": email_text,
                    "result": result,
                    "type": 'CheckedEmail'    
                }

                print(json_data)
                await WebSocketHandler.send_message(websocket, json_data)
            elif 'url' in data:
                url = data['url']
                    
                url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable]]
                X = sequence.pad_sequences(url_int_tokens, maxlen=MAX_LEN)

                if data['type'] == 'PreCheck' or data['type'] == 'CurrentTab':
                    target_proba = url_classifier_model.predict(X, batch_size=1)
                    result = "malicious" if target_proba > 0.5 else "benign"

                    # Get the domain
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc

                    # Get the IP address associated with the domain
                    ip_address = socket.gethostbyname(domain)

                    # Getting IP information
                    if ip_address != '0.0.0.0':
                        ip_info = IPWhois(ip_address).lookup_whois()
                        country = ip_info["asn_country_code"]
                        organization = ip_info["asn_description"]
                    else:
                        country = "Reserved"
                        organization = "Reserved"

                    # Getting product information
                    headers = requests.get(url).headers
                    product = headers.get("Server", "Unknown")

                    json_data = {
                        "url": url,
                        "target_proba": float(target_proba),
                        "result": result,
                        "domain": domain,
                        "ip_address": ip_address,
                        "country": country,
                        "organization": organization,
                        "product": product,
                        "type": data['type']    
                    }
                    
                    print(json_data)
                    await WebSocketHandler.send_message(websocket, json_data)
                elif data['type'] == 'FalseNegative':
                    df = pd.read_csv(PATH_TO_URL_DATASET)
                    df["isMalicious"] = df["Label"].replace({
                        'Benign': 0,
                        'Phishing': 1,
                    })
                    RetrainUtils.save_misclassified_urls(url, 0 if data['result'] == 'phishing' else 1)

                    false_negatives = pd.read_csv(PATH_TO_MISCLASSIFIED_URLS)

                    if len(false_negatives) >= MAX_LEN_MISCLASSIFIED_URLS:
                        url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df.URLs]
                        X = sequence.pad_sequences(url_int_tokens, maxlen=MAX_LEN)
                        target = np.array(df.isMalicious)
                        
                        X_train, X_test, target_train, target_test = model_selection.train_test_split(X, target, test_size=TEST_SIZE, random_state=RANDOM_STATE)

                        model = LSTMClassifier().build_model();
                        await RetrainUtils.train_with_misclassified_urls(model, X_train, target_train, X_test, target_test, false_negatives, websocket)
                        
                        save_model('phishing_url_detector/models/' + URL_MODEL_NAME + ".json",
                                   'phishing_url_detector/models/' + URL_MODEL_NAME + ".h5", model)
                        
                    json_data = {
                    "type": 'RetrainFinished'   
                    }
                    
                    await WebSocketHandler.send_message(websocket, json_data)
            elif 'urls' in data and data['type'] == 'FetchUrls':
                urls = data['urls']
                analyzed_urls = []
                for url in urls:
                    url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable]]
                    X = sequence.pad_sequences(url_int_tokens, maxlen=MAX_LEN)

                    target_proba = url_classifier_model.predict(X, batch_size=1)
                    result = "malicious" if target_proba > 0.5 else "benign"
                    
                    url_data = {
                        "url": url,
                        "result": result
                    }
                    analyzed_urls.append(url_data)
                    
                json_data = {
                    "urls": analyzed_urls,
                    "type": "AnalyzedFetchedUrls"
                }
                print(json_data)
                await WebSocketHandler.send_message(websocket, json_data)

