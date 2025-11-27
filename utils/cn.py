from utils.tensor import Tensor
from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

class Parameter(Tensor):
    def __init__(self, val: np.ndarray|list, grad_fn=lambda: [], custom_name=None):
        super().__init__(val, grad_fn, custom_name)
        self.m = 0
        self.vel = 0

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
    def __init__(self, n_in:int, n_out:int, bias=False):
        super().__init__()
        k = 1 / n_in
        self.parameter = Parameter(np.random.uniform(-np.sqrt(k), np.sqrt(k), size=[n_in, n_out]))
        self.bias = Parameter(np.random.uniform(-np.sqrt(k), np.sqrt(k), size=[n_out]))

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.parameter + self.bias
        return x