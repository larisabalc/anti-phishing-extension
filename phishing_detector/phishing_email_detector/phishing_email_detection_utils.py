import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from transformers import PreTrainedTokenizerBase

def load_data(data_file: str) -> tuple:
    """
    Load data from a CSV file.

    Args:
        data_file (str): Path to the CSV file containing the data.

    Returns:
        tuple: A tuple containing texts and labels.
    """
    df = pd.read_csv(data_file)
    texts = df['body'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def train(model: nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, device: torch.device):
    """
    Train the model.

    Args:
        model (nn.Module): The model to be trained.
        data_loader (DataLoader): DataLoader for loading batches of data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler for controlling learning rate.
        device (torch.device): Device to run the training on.
    """
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> tuple:
    """
    Evaluate the model.

    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for loading batches of data.
        device (torch.device): Device to run the evaluation on.

    Returns:
        tuple: A tuple containing accuracy score and classification report.
    """
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

def predict_is_phishing(text: str, model: nn.Module, tokenizer: PreTrainedTokenizerBase, device: torch.device, max_length: int = 128) -> str:
    """
    Predict if a given text is phishing or benign.

    Args:
        text (str): The input text to be classified.
        model (nn.Module): The trained model.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for tokenizing the input text.
        device (torch.device): Device to run the prediction on.
        max_length (int): Maximum length of the input text.

    Returns:
        str: "phishing" if the text is predicted as phishing, otherwise "benign".
    """
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    print(encoding)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return "phishing" if preds.item() == 1 else "benign"
