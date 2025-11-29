from utils.tensor import Tensor
from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

class Parameter(Tensor):
    def __init__(self, val: np.ndarray|list, grad_fn=lambda: [], custom_name=None, requires_grad=True):
        super().__init__(val, grad_fn, custom_name, requires_grad=requires_grad)
        self.m = 0
        self.vel = 0

class Module(ABC):
    def __init__(self):
        self.train_mode = True #NOTE: Not used atm, but needs to be used when we implement batchnorm

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

    def eval(self):
        self.train_mode = False
    
    def train(self):
        self.train_mode = True


class Sicquential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class Flatten(Module):
    def __init__(self, start_dim:int=1, end_dim:int=-1): #By default doesn't flatten the first dim (the batch dim)
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x:Tensor) -> Tensor:
        return x.flatten(start_dim=self.start_dim, end_dim=self.end_dim)

class Unflatten(Module):
    def __init__(self, unflatten_dim: int = 1, shape: tuple[int, ...] = ()):
        super().__init__()
        self.unflatten_dim = unflatten_dim
        self.shape = shape

    def forward(self, x:Tensor) -> Tensor:
        return x.unflatten(unflatten_dim=self.unflatten_dim, shape=self.shape)

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class Linear(Module):
    def __init__(self, n_in:int, n_out:int, bias=False):
        super().__init__()
        k = 1 / n_in
        self.parameter = Parameter(np.random.uniform(-np.sqrt(k), np.sqrt(k), size=[n_in, n_out])) # TODO: Move this and bias into an initializer
        self.bias = Parameter(np.random.uniform(-np.sqrt(k), np.sqrt(k), size=[n_out]))

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.parameter + self.bias
        return x

class Conv2D(Module):
    def __init__(self, in_channels: int, num_kernels: int, kernel_size: tuple[int, int], stride: int = 1, padding: int | str = 0, bias: bool = True, padding_mode: str = 'zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_kernels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        kernel_shape = (num_kernels, in_channels, self.kernel_size[0], self.kernel_size[1])
        k = 1 / (in_channels * self.kernel_size[0] * self.kernel_size[1])
        self.kernels = Parameter(np.random.uniform(-np.sqrt(k), np.sqrt(k), size=kernel_shape))
        self.stride = stride
        if isinstance(padding, str):
            if padding != 'same':
                raise ValueError('Padding must be "same" or int')
            if self.kernel_size[0] != self.kernel_size[1]:
                raise ValueError(f'Padding can only be "same" if kernel is square. Kernel is {self.kernel_size[0]} by {self.kernel_size[1]}')
            self.padding
            self.padding = self.kernel_size[0] // 2
        else:
            self.padding = padding
        self.use_bias = bias
        self.padding_mode = padding_mode
    
    def pad(self, x:Tensor) -> Tensor:
        x = 

    def img2col(self, x:Tensor, stride:int):
        im_flat = np.zeros([self.kernels.v.size, x.v.size])
        h_out = int((x.v.shape[2] + 2 * self.padding - self.kernel_size[0]) // self.stride + 1)
        w_out = int((x.v.shape[3] + 2 * self.padding - self.kernel_size[1]) // self.stride + 1)
        batches = int(x.v.shape[0])
        channels = self.in_channels

        for i in range(self.kernels.v.size * channels):
            for j in range(h_out * w_out * batches):
                img_nr = j // batches
                ch = i // (h_out * w_out)
                axis_0 = i // (self.kernel_size[0] * (ch+1))
                axis_1 = j // (self.kernel_size[1] * (img_nr+1))
                im_flat[i,j] = x.v[img_nr, ch, axis_0, axis_1]

        im_flat = Tensor(im_flat)
        return im_flat
    
