# Copy and pasted from https://github.com/rasmusbergpalm/nanograd/blob/3a1bf9e9e724da813bfccf91a6f309abdade9f39/nanograd.py

from math import exp, log
import numpy as np

class Tensor:
    """
    A tensor which holds a array and enables gradient computations.
    """

    def __init__(self, val: np.ndarray|list, grad_fn=lambda: []):
        #assert type(val) == np.ndarray
        self.v = val
        self.grad_fn = grad_fn
        self._grad = None

        if type(self.v) != np.ndarray:
            self.v = np.array(self.v)

    def backprop(self, bp):
        if self._grad == None:
            self._grad = np.zeros(self.v.shape)

        self._grad += bp

        for input, grad in self.grad_fn():
            if "T" in grad:
                input.backprop(bp.T)
            else:
                #print(grad, bp)
                input.backprop(grad * bp)

    @property
    def T(self):
        return Tensor(self.v.T, lambda: [(self, np.array(["T"]))])

    def backward(self):
        self.backprop(1.0)

    def __add__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        #assert self.v.shape != other.v.shape ""
        return Tensor(self.v + other.v, lambda: [(self, np.ones(self.v.shape)), (other, np.ones(other.v.shape))])
    
    def __mul__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        return Tensor(self.v * other.v, lambda: [(self, other.v), (other, self.v)])

    def __matmul__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        return Tensor(self.v @ other.v, lambda: [(self, other.v.T), (other, self.v.T)])

    def __pow__(self, power):
        assert type(power) in {float, int}, "power must be float or int"
        return Tensor(self.v ** power, lambda: [(self, power * self.v ** (power - 1))])

    def __neg__(self: 'Tensor') -> 'Tensor':
        return Tensor(-1.0) * self

    def __sub__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        return self + (-other)

    def __truediv__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        return self * other ** -1

    def grad(self):
        return self._grad

    def __repr__(self):
        return str(self.v) #"Var(v=%.4f, grad=%.4f)" % (self.v, self.grad) if self.grad != None else "Var(v=%.4f, grad=None)" % (self.v)

    def relu(self):
        return Tensor(self.v if self.v > 0.0 else 0.0, lambda: [(self, 1.0 if self.v > 0.0 else 0.0)])
    
    def zerograd(self,grad_none=False):
        if grad_none:
            self._grad = None
        else:
            self._grad = 0.0

if __name__ == "__main__":
    a = Tensor(np.array([2, 3]))
    b = Tensor(np.array([2, 4]).T)

    f = b @ a
    f.backward()
    print(f)
    print(a.grad())

    # Should be equal to the above
    a = Tensor(np.array([2, 3]))
    b = Tensor(np.array([2, 4])).T

    f = b @ a
    f.backward()
    print(f)
    print(a.grad())
