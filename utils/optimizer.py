# from utils.tensor import Tensor
# from denselayer import DenseLayer
# import numpy as np
# from loss_function import cross_entropy_loss
# from initializer import NormalInitializer
from abc import ABC, abstractmethod
import numpy as np

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
    def __init__(self, params, lr=0.001, weight_decay=0.0):
        super().__init__(params, lr)
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if param.grad is not None:
                grad = param.grad + self.weight_decay * param.v
                param.v = param.v - self.lr * grad

class SGDMomentum(Optimizer):
    def __init__(self, params, lr=0.001, beta = 0.9, weight_decay=0.0):
        super().__init__(params, lr)
        self.beta = beta
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if param.grad is not None:
                grad = param.grad + self.weight_decay * param.v
                param.m = param.m * self.beta + (1-self.beta) * grad
                param.v = param.v - param.m * self.lr

class ADAM(Optimizer):
    def __init__(self, params, lr=0.001, beta = 0.9, gamma = 0.999, weight_decay=0.0):
        super().__init__(params, lr)
        self.beta = beta
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.t = 0
        self.eps = 1.e-8


    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue
            
            grad = param.grad + self.weight_decay * param.v
            param.m =  param.m * self.beta + (1-self.beta) * grad
            param.vel = param.vel * self.gamma + (1-self.gamma) * grad**2

            m_tilde = param.m / (1 - self.beta**self.t)
            vel_tilde = param.vel / (1 - self.gamma**self.t)

            param.v = param.v - self.lr * m_tilde /(np.sqrt(vel_tilde) + self.eps)


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
