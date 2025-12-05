from utils.cn import Flatten, Linear, Module, ReLU, Sequential, Sigmoid

class FFNN(Module):
    def __init__(self, num_classes:int, in_channels:int, hidden_layer_config:list[int]=[600, 600, 120], activation:str='ReLU', initializer:str='Normal'):
        super().__init__()

        if activation == 'ReLU':
            activation_fn = ReLU
        elif activation == 'Sigmoid':
            activation_fn = Sigmoid
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = [Flatten()]

        # Build hidden layers based on config
        prev_size = in_channels
        for hidden_size in hidden_layer_config:
            layers.append(Linear(n_in=prev_size, n_out=hidden_size, initializer=initializer))
            layers.append(activation_fn())
            prev_size = hidden_size
        
        # Output layer
        layers.append(Linear(n_in=prev_size, n_out=num_classes, initializer=initializer))
        
        self.layers = Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)