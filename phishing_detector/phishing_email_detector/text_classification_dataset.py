from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    """
    Dataset for text classification tasks.

    This dataset class is designed to work with text classification tasks. It takes a list of text samples
    along with their corresponding labels and performs tokenization using a provided tokenizer.

    Args:
        texts (List[str]): List of text samples.
        labels (List[int]): List of corresponding labels.
        tokenizer (Any): Tokenizer object.
        max_length (int): Maximum length of the input sequence after tokenization.
    
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns a sample from the dataset at the given index.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: Any, max_length: int):
        """
        Initializes the dataset.

        Args:
            texts (List[str]): List of text samples.
            labels (List[int]): List of corresponding labels.
            tokenizer (Any): Tokenizer object.
            max_length (int): Maximum length of the input sequence after tokenization.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing input_ids, attention_mask, and label tensors.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
