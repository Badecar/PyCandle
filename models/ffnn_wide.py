from models.ffnn import FFNN

class FFNN_Wide(FFNN):
    """Wider FFNN architecture with more neurons in middle layers: 800 -> 1000 -> 800 -> 200"""
    def __init__(self, num_classes:int, in_channels:int, activation:str='ReLU', initializer:str='Normal'):
        super().__init__(num_classes, in_channels, hidden_layer_config=[800, 1000, 800, 200], activation=activation, initializer=initializer)