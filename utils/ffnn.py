import numpy as np
from typing import Sequence

from denselayer import DenseLayer
from utils.tensor import Tensor
from utils.initializer import NormalInitializer, ConstantInitializer


class FFNN:
    def __init__(self, in_channels:int, num_classes:int, num_hidden_layers:int, hidden_dims:Sequence, act_fns:Sequence, batch_size:int, initializer=NormalInitializer()):
        self.dims = np.append(in_channels, hidden_dims, num_classes)
        self.layers = []
        for i in range(num_hidden_layers):
            l = DenseLayer(n_in=self.dims[i], n_out=self.dims[i+1], batch_size=batch_size, act_fn=act_fns[i], initializer=initializer)
            self.layers.append(l)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

