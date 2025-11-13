# Copy and pasted from https://github.com/rasmusbergpalm/nanograd/blob/3a1bf9e9e724da813bfccf91a6f309abdade9f39/nanograd.py

from math import exp, log
from typing import Sequence
import numpy as np
from numpy.linalg import tensorsolve

class Tensor:
    """
    A tensor which holds a array and enables gradient computations.
    """

    def __init__(self, val: np.ndarray|list, grad_fn=lambda: [], custom_name=None):
        #assert type(val) == np.ndarray
        self.v = np.array(val, dtype=float)
        self.grad_fn = grad_fn
        self._grad = None
        self.custom_name = custom_name

        if type(self.v) != np.ndarray:
            self.v = np.array(self.v)

    @property
    def T(self):
        return Tensor(self.v.T, lambda: [(self, np.array(["T"]))])

    def backprop(self, bp: Tensor):
        print("debug: backprop on", self.custom_name)
        if self._grad is None:
            self._grad = np.zeros_like(self.v)

        self._grad += bp

        for grad_fn in self.grad_fn():
            input: Tensor = grad_fn["input"]
            grad: np.ndarray = grad_fn["grad"]
            try:
                start_index = grad_fn["start_index"]
                end_index = grad_fn["end_index"]
            except:
                start_index = None
                end_index = None

            if "T" in grad:
                input.backprop(bp.T)
            elif start_index is not None and end_index is not None:
                input.backprop(grad * bp[start_index:end_index])
            else:
                input.backprop(grad * bp)

    def backward(self):
        self.backprop(np.ones_like(self.v))

    def cat(self, others: Sequence['Tensor'], axis: int):
        all = [self] + others
        current_index = 0
        outputs = []
        for part in all:
            output = {
                "input": part,
                "grad": np.ones_like(part.v),
                "start_index": current_index,
                "end_index": current_index + part.v.shape[axis]
            }
            outputs.append(output)
            current_index += part.v.shape[axis]
        return Tensor(np.concat([o.v for o in all], axis=axis), lambda: outputs, custom_name=f"cat({', '.join([o.custom_name for o in all])})")

    def __add__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        #assert self.v.shape != other.v.shape ""
        return Tensor(self.v + other.v, lambda: [{"input": self, "grad": np.ones(self.v.shape)}, {"input": other, "grad": np.ones(other.v.shape)}], custom_name=f"{self.custom_name} + {other.custom_name}")
    
    def __mul__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        return Tensor(self.v * other.v, lambda: [{"input": self, "grad": other.v}, {"input": other, "grad": self.v}], custom_name=f"{self.custom_name} * {other.custom_name}")

    def __matmul__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        return Tensor(self.v @ other.v, lambda: [{"input": self, "grad": other.v.T @ np.ones_like(self.v)}, {"input": other, "grad": self.v.T @ np.ones_like(other.v)}])

    def __pow__(self, power):
        assert type(power) in {float, int}, "power must be float or int"
        return Tensor(self.v ** power, lambda: [{"input": self, "grad": power * self.v ** (power - 1)}])

    def __neg__(self: 'Tensor') -> 'Tensor':
        return Tensor(-1.0) * self

    def __sub__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        return self + (-other)

    def __truediv__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        return self * other ** -1

    def grad(self):
        return self._grad

    def __repr__(self):
        output_string = ""
        if self.custom_name is not None:
            output_string += self.custom_name + " "
        output_string += str(self.v)
        if self._grad is not None:
            output_string += " grad: " + str(self._grad)
        return output_string


    def relu(self):
        return Tensor(np.maximum(self.v, 0.0), lambda: [(self, (self.v > 0.0).astype(float))])
    
    def zerograd(self,grad_none=False):
        if grad_none:
            self._grad = None
        else:
            self._grad = 0.0

if __name__ == "__main__":
    a = Tensor(np.array([1,2,3]))
    b = Tensor(np.array([[-1,-2,-3],[4,5,6],[7,8,9]]))
    bias = Tensor(np.ones(3)*2)
    print(a)
    print(b)
    f = b @ a
    print(f)
    # f = f.relu()
    print(f)

    f.backward()

    print(a.grad())
    # print(b.grad())
    # print(bias.grad())
    a = Tensor(np.array([1],dtype=float), custom_name="a")
    b = Tensor(np.array([2],dtype=float), custom_name="b")
    c = Tensor(np.array([1],dtype=float), custom_name="c")
    d = Tensor(np.array([2],dtype=float), custom_name="d")
    e = Tensor(np.array([3, 3],dtype=float), custom_name="e")

    ab = a * b
    cd = c * d

    f = ab.cat([cd], 0)

    g = f @ e
    g.backward()
    print(f.grad())
    print(ab.grad())
    print(cd.grad())
    print(f.grad())
    print(e.grad())
    
    
    
