from utils.tensor import Tensor
from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

class Parameter(Tensor):
    def __init__(self, val: np.ndarray|list, grad_fn=lambda: [], custom_name=None):
        super().__init__(val, grad_fn, custom_name)

class Module(ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self):
        params = []
        for name, attr in self.__dict__.items():
            if isinstance(attr, Parameter):
                params.append(attr)
            if isinstance(attr, Module):
                params.extend(attr.parameters())
            if isinstance(attr, Sequence) and all(isinstance(item, Module) for item in attr):
                for item in attr:
                    params.extend(item.parameters())
                
        return params

class Sicquential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        x = x.relu()
        return x

class Linear(Module):
    def __init__(self, n_in:int, n_out:int, initializer=None):
        super().__init__()
        if initializer == None:
            from utils.initializer import NormalInitializer
            initializer = NormalInitializer()
        self.initializer = initializer
        self.parameter = self.initializer.init_weights(n_in=n_in, n_out=n_out)
        self.bias = Parameter(np.zeros([32, n_out])) #TODO: Fix this with broadcasting

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.parameter + self.bias
        return x