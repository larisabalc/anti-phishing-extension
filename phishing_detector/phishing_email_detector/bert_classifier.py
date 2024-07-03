from torch import nn
from torch import Tensor
from transformers import BertModel

class BERTClassifier(nn.Module):
    """
    BERT-based classifier model.

    This classifier utilizes a pre-trained BERT model for feature extraction and a bidirectional GRU layer for classification.

    Attributes:
        bert (BertModel): Pre-trained BERT model.
        dropout (nn.Dropout): Dropout layer for regularization.
        bigru (nn.GRU): Bidirectional GRU layer for classification.
        fc (nn.Linear): Fully connected layer for output classification.
    
    Methods:
        forward(input_ids, attention_mask): Performs a forward pass of the model.
    """
    
    def __init__(self, bert_model_name: str, num_classes: int, hidden_size: int = 128, num_layers: int = 1):
        """
        Initializes the BERTClassifier.

        Args:
            bert_model_name (str): Name of the pre-trained BERT model.
            num_classes (int): Number of classes for classification.
            hidden_size (int, optional): Size of the hidden layer. Defaults to 128.
            num_layers (int, optional): Number of GRU layers. Defaults to 1.
        """
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.bigru = nn.GRU(self.bert.config.hidden_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (Tensor): Input token ids.
            attention_mask (Tensor): Attention mask to avoid performing attention on padding tokens.

        Returns:
            Tensor: Output logits.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = pooled_output.unsqueeze(0)  # Add batch dimension
        bigru_output, _ = self.bigru(pooled_output)
        bigru_output = bigru_output.squeeze(0)  # Remove batch dimension
        x = self.dropout(bigru_output)
        logits = self.fc(x)
        return logits
