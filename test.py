
from utils.cn import Linear
from utils.tensor import Tensor
import numpy as np

m = Linear(n_in=600, n_out=600, bias=True)

x = Tensor(np.random.randn(10, 600))

y = m(x)
y.backward()

print(y)
print(m.parameter.grad)
print(m.bias.grad)