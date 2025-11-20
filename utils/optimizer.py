from tensor import Tensor
from denselayer import DenseLayer
import numpy as np
from loss_function import cross_entropy_loss
from initializer import NormalInitializer

class Optimizer():

    def SGD(model, lr=0.001,T_end=1000):
        


        for t in range(1,T_end):
            print(t)
            gt = params[t].grad * params[t]
            params[t] = params[t] - lr * gt

        return params[t]

if __name__ == "__main__":
    
    input_ = Tensor(np.array([[1, 2],
                            [3, 4],
                            [5, 6]]))
    batch_size = input_.v.shape[1]

    # layer = DenseLayer(n_in=3, n_out=3, batch_size=batch_size, act_fn=Tensor.relu, initializer=ConstantInitializer(weight=2, bias=2))
    layer = DenseLayer(n_in=3, n_out=3, batch_size=batch_size, act_fn=Tensor.relu, initializer=NormalInitializer())
    params = layer.parameters()
    print('params:' , params)
    optimizer = SGD(params, lr=0.1)
    optimizer.zero_grad()
    loss_fn = cross_entropy_loss
    loss_fn.backward()
    optimizer.step()