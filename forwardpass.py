import numpy as np
from typing import Sequence

from utils.var import Tensor
from utils.initializer import NormalInitializer, ConstantInitializer

class DenseLayer:
    def __init__(self, n_in: int, n_out: int, act_fn, initializer = NormalInitializer()):
        self.weights = initializer.init_weights(n_in, n_out)
        self.bias = initializer.init_bias(n_out)
        self.act_fn = act_fn  #activation function
    
    def __repr__(self):    
        return 'Weights: ' + repr(self.weights) + ' Biases: ' + repr(self.bias)

    def parameters(self) -> Sequence[Tensor]:
      params = []
      for r in self.weights:
        params.append(r)

      return params + self.bias

    def forward(self, x: Sequence[Tensor]) -> Sequence[Tensor]:
        # self.weights is a matrix with dimension n_in x n_out. We check that the dimensionality of the input 
        # to the current layer matches the number of nodes in the current layer
        weights = self.weights
        out = []
        # For some given data point single_input, we now want to calculate the resulting value in each node in the current layer
        # We therefore loop over the (number of) nodes in the current layer:
        out = weights @ x + self.bias
        
        

        # for j in range(len(weights[0])): 
        #     # Initialize the node value depending on its corresponding parameters.
        #     node = Var(0.0) # <- Insert code
        #     # We now finish the linear transformation corresponding to the parameters of the currently considered node.
        #     for i in range(len(single_input)):
        #         node += Var(0.0)  # <- Insert code
        #     node = self.act_fn(node)
        #     out.append(node)

        return out


layer = DenseLayer(3,3,act_fn=Tensor.relu,initializer=NormalInitializer())

input_ = np.array([Tensor(1.),Tensor(2.),Tensor(3.)])
print(layer.parameters())
y = layer.forward(input_)
for i in y:
   i.backward()

print("____")
print(layer.parameters())


