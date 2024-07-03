import os
import json
from pathlib import Path
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model

def print_layers_dims(model: Model) -> None:
    """
    Print the input and output shapes of each layer in the model.
    """
    l_layers = model.layers

    for i in range(len(l_layers)):
        print(l_layers[i])
        print('Input Shape: ', l_layers[i].input_shape, 'Output Shape: ', l_layers[i].output_shape)

def save_model(fileModelJSON: str, fileWeights: str, model: Model) -> None:
    """
    Save the model architecture as JSON and weights to separate files.
    """
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    
    json_string = model.to_json()
    with open(fileModelJSON, 'w') as f:
        json.dump(json_string, f)
    
    model.save_weights(fileWeights)

def load_model(fileModelJSON: str, fileWeights: str) -> Model:
    """
    Load model architecture from JSON and load weights from file.
    """
    with open(fileModelJSON, 'r') as f:
        model_json = json.load(f)
        model = model_from_json(model_json)
    
    model.load_weights(fileWeights)
    return model