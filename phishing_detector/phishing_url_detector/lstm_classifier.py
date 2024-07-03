from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

class LSTMClassifier:
    """
    Simple LSTM Model

    This class implements a simple LSTM model for binary classification tasks.

    Args:
        max_len (int): Maximum length of input sequences (default: 75).
        emb_dim (int): Dimensionality of the embedding space (default: 32).
        max_vocab_len (int): Maximum size of the vocabulary (default: 100).
        lstm_output_size (int): Number of units in the LSTM layer (default: 32).
        reg (Regularizer): Regularizer for the embedding layer (default: L2 regularization with 1e-4).
    
    Methods:
        build_model(): Builds and compiles the LSTM model.
    """
    
    def __init__(self, max_len: int = 75, emb_dim: int = 32, max_vocab_len: int = 100, lstm_output_size: int = 32, reg=regularizers.l2(1e-4)):
        """
        Initializes the SimpleLSTM object.

        Args:
            max_len (int): Maximum length of input sequences (default: 75).
            emb_dim (int): Dimensionality of the embedding space (default: 32).
            max_vocab_len (int): Maximum size of the vocabulary (default: 100).
            lstm_output_size (int): Number of units in the LSTM layer (default: 32).
            reg (Regularizer): Regularizer for the embedding layer (default: L2 regularization with 1e-4).
        """
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.max_vocab_len = max_vocab_len
        self.lstm_output_size = lstm_output_size
        self.reg = reg
        
        self.main_input: Input = Input(shape=(self.max_len,), dtype='int32', name='main_input')
        self.emb = Embedding(input_dim=self.max_vocab_len, output_dim=self.emb_dim, input_length=self.max_len, embeddings_regularizer=self.reg)(self.main_input)
        self.lstm = LSTM(self.lstm_output_size)(self.emb)
        self.dropout = Dropout(0.5)(self.lstm)
        self.output = Dense(1, activation='sigmoid', name='output')(self.dropout)

    def build_model(self) -> Model:
        """
        Builds and compiles the LSTM model.

        Returns:
            Model: The compiled Keras model.
        """
        model: Model = Model(inputs=[self.main_input], outputs=[self.output])
        adam: Adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        return model
