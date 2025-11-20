from utils.tensor import Tensor
from abc import ABC, abstractmethod
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
        return params