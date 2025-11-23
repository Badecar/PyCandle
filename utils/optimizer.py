# from utils.tensor import Tensor
# from denselayer import DenseLayer
# import numpy as np
# from loss_function import cross_entropy_loss
# from initializer import NormalInitializer
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, params, lr=0.001):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.v = param.v - self.lr * param.grad

# if __name__ == "__main__":
    
#     input_ = Tensor(np.array([[1, 2],
#                             [3, 4],
#                             [5, 6]]))
#     batch_size = input_.v.shape[1]

#     # layer = DenseLayer(n_in=3, n_out=3, batch_size=batch_size, act_fn=Tensor.relu, initializer=ConstantInitializer(weight=2, bias=2))
#     layer = DenseLayer(n_in=3, n_out=3, batch_size=batch_size, act_fn=Tensor.relu, initializer=NormalInitializer())
#     params = layer.parameters()
#     print('params:' , params)
#     optimizer = SGD(params, lr=0.1)
#     optimizer.zero_grad()
#     loss_fn = cross_entropy_loss
#     loss_fn.backward()
#     optimizer.step()
