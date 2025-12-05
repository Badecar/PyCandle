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
        self.train_mode = True #NOTE: Not used atm, but needs to be used if we implement Batchnorm

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


class Sequential(Module):
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

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

class Linear(Module):
    def __init__(self, n_in:int, n_out:int, bias=True, initializer='Uniform'):
        super().__init__()
        
        # Handle both string and object initializers
        if isinstance(initializer, str):
            # Import here to avoid circular import
            from utils.initializer import NormalInitializer, UniformInitializer
            if initializer == 'Normal':
                initializer_fn = NormalInitializer()
            elif initializer == 'Uniform':
                initializer_fn = UniformInitializer()
            else:
                raise ValueError(f"Unknown initializer: {initializer}")
        else:
            # Assume it's already an initializer object
            initializer_fn = initializer

        self.parameter = initializer_fn.init_weights(n_in, n_out)
        self.bias = initializer_fn.init_bias(n_out) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        if self.bias is not None:
            x = x @ self.parameter + self.bias
        else:
            x = x @ self.parameter
        return x

class Conv2D(Module):
    def __init__(self, in_channels: int, num_kernels: int, kernel_size: int|tuple[int, int], stride: int = 1, padding: int | str = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_kernels
        self.stride = stride
        self.use_bias = bias

        # Defining kernels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        kernel_shape = (num_kernels, in_channels, self.kernel_size[0], self.kernel_size[1])
        k = 1 / (in_channels * self.kernel_size[0] * self.kernel_size[1])
        self.kernels = Parameter(np.random.uniform(-np.sqrt(k), np.sqrt(k), size=kernel_shape))

        # Defining bias
        if self.use_bias:
            #(Out_Channels, 1, 1) to broadcast over (N, Out_C, H, W)
            self.bias = Parameter(np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(num_kernels, 1, 1)))
        else:
            self.bias = None
        
        # Defining padding
        if isinstance(padding, str):
            if padding != 'same':
                raise ValueError('Padding must be "same" or int')
            if self.kernel_size[0] != self.kernel_size[1]:
                raise ValueError(f'Padding can only be "same" if kernel is square. Kernel is {self.kernel_size[0]} by {self.kernel_size[1]}')
            self.padding = self.kernel_size[0] // 2
        else:
            self.padding = padding

    def forward(self, x:Tensor) -> Tensor:
        col, (N, h_out, w_out) = x.img2col(
            stride=self.stride,
            kernels=self.kernels,
            padding=self.padding,
            kernel_size=self.kernel_size,
            in_channels=self.in_channels,
        )
        
        kernel_matrix = self.kernels.flatten(start_dim=1) #(Out_C, In_C * KH * KW)
        output = kernel_matrix @ col #(Out_C, Pixels)
        output = output.unflatten(1, (N, h_out, w_out)) #(Out_C, N, H_out, W_out)
        
        output = output.permute(1, 0, 2, 3) # (N, Out_C, H_out, W_out)

        if self.use_bias:
            output = output + self.bias
        return output #Boom!


class MaxPool2D(Module):
    def __init__(self, kernel_size: int | tuple[int, int], stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size[0]
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return x.max_pool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )