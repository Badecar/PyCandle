from models.ffnn import FFNN

class FFNN_Standard(FFNN):
    """Standard FFNN architecture: 600 -> 600 -> 120"""
    def __init__(self, num_classes:int, in_channels:int, activation:str='ReLU', initializer:str='Normal'):
        super().__init__(num_classes, in_channels, hidden_layer_config=[600, 600, 120], activation=activation, initializer=initializer)