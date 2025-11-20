import numpy as np
from typing import Sequence

from utils.tensor import Tensor
from utils.initializer import NormalInitializer, ConstantInitializer

class DenseLayer:
    def __init__(self, n_in: int, n_out: int, batch_size: int, act_fn, initializer = NormalInitializer()):
        self.weights = initializer.init_weights(n_in, n_out)
        self.bias = initializer.init_bias(n_out, batch_size=batch_size)
        self.act_fn = act_fn  #activation function

    def __repr__(self):    
        return 'Weights: ' + repr(self.weights) + ' Biases: ' + repr(self.bias)

    def parameters(self) -> Sequence[Tensor]:
      params = []
      for r in self.weights.v:
        params.append(r)

      return np.append(params, self.bias.v)

    def forward(self, x: Tensor) -> Tensor:
        # self.weights is a matrix with dimension n_in x n_out. We check that the dimensionality of the input 
        
        # print("weights shape:", self.weights.v.shape)
        # print("input shape:", x.v.shape)
        # print("bias shape:", self.bias.v.shape)
        out = self.weights @ x + self.bias
        out = self.act_fn(out)
        return out



        # for j in range(len(weights[0])): 
        #     # Initialize the node value depending on its corresponding parameters.
        #     node = Var(0.0) # <- Insert code
        #     # We now finish the linear transformation corresponding to the parameters of the currently considered node.
        #     for i in range(len(single_input)):
        #         node += Var(0.0)  # <- Insert code
        #     node = self.act_fn(node)
        #     out.append(node)

        return out


input_ = Tensor(np.array([[1, 2],
                         [3, 4],
                         [5, 6]]))
batch_size = input_.v.shape[1]

# layer = DenseLayer(n_in=3, n_out=3, batch_size=batch_size, act_fn=Tensor.relu, initializer=ConstantInitializer(weight=2, bias=2))
layer = DenseLayer(n_in=3, n_out=3, batch_size=batch_size, act_fn=Tensor.relu, initializer=NormalInitializer())


y = layer.forward(input_)
print(y)
y.backward()


# print("____")
# print(layer.parameters())


