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
    
    @property
    def grad(self):
        return self._grad

    def backprop(self, bp, debug=False):
        if self._grad is None:
            self._grad = np.zeros_like(self.v)

        self._grad += bp
        for grad_fn in self.grad_fn():
            input: Tensor = grad_fn["input"]
            grad: np.ndarray = grad_fn["grad"]

            start_index = grad_fn.get("start_index", None)
            end_index = grad_fn.get("end_index", None)
            matrix_side = grad_fn.get("matrix", None)

            if debug:
                print("custom_name", self.custom_name)
                print("bp", bp)
                print("grad", grad)

            if matrix_side != None:
                if matrix_side == "L":
                    input.backprop(bp @ grad)
                elif matrix_side == "R":
                    input.backprop(grad @ bp)
                else:
                    raise ValueError(f"Invalid matrix_side value: {matrix_side}")

            elif "T" in grad:
                input.backprop(bp.T)
            # CAT PART OF BACKPROP NOT FIXED YET!!!
            elif start_index is not None and end_index is not None:
                input.backprop(grad @ bp[start_index:end_index])
            else:
                input.backprop(grad * bp)

    def backward(self, debug=False):
        # self.backprop(np.array([1]))
        self.backprop(np.ones_like(self.v), debug=debug)

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
        return Tensor(np.concatenate([o.v for o in all], axis=axis), lambda: outputs, custom_name=f"cat({', '.join([o.custom_name for o in all])})")

    def __add__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        #assert self.v.shape != other.v.shape ""
        return Tensor(self.v + other.v, lambda: [{"input": self, "grad": np.ones_like(self.v)}, {"input": other, "grad": np.ones_like(other.v)}], custom_name=f"({self.custom_name} + {other.custom_name})")
    
    def __mul__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        return Tensor(self.v * other.v, lambda: [{"input": self, "grad": other.v}, {"input": other, "grad": self.v}], custom_name=f"({self.custom_name} * {other.custom_name})")

    def __matmul__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        if len(self.v.shape) == 1 and len(other.v.shape) == 1:
            return self.__mul__(other)
        if self.v.shape[-1] != other.v.shape[-2]:
            raise ValueError(f"Shape mismatch: {self.v.shape} @ {other.v.shape}")
        return Tensor(self.v @ other.v, lambda: [{"input": self, "grad": other.v.T, "matrix": "L"}, {"input": other, "grad": self.v.T, "matrix": "R"}], custom_name=f"({self.custom_name} @ {other.custom_name})")

    def __pow__(self, power):
        assert type(power) in {float, int}, "power must be float or int"
        return Tensor(self.v ** power, lambda: [{"input": self, "grad": power * self.v ** (power - 1)}], custom_name=f"({self.custom_name} ** {power})")

    def __neg__(self: 'Tensor') -> 'Tensor':
        result = Tensor(-1.0) * self
        result.custom_name = f"-({self.custom_name})"
        return result

    def __sub__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        return self + (-other)

    def __truediv__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        return self * other ** -1

    def __repr__(self):
        output_string = ""
        if self.custom_name is not None:
            output_string += self.custom_name + " "
        output_string += str(self.v)
        if self._grad is not None:
            output_string += " grad: " + str(self._grad)
        return output_string
    
    def sum(self, axis=None, keepdims=False):
        result = np.sum(self.v, axis=axis, keepdims=keepdims)
        return Tensor(result, lambda: [{"input": self, "grad": np.ones_like(self.v)}])
    
    def exp(self):
        ev = np.exp(self.v)
        return Tensor(ev, lambda: [(self, ev)])

    def log(self):
        return Tensor(np.log(self.v), lambda: [(self, self.v ** -1)])
    def relu(self):
        return Tensor(np.maximum(self.v, 0.0), lambda: [{"input": self, "grad": (self.v > 0.0).astype(float)}])
    
    def zero_grad(self,grad_none=False):
        if grad_none:
            self._grad = None
        else:
            self._grad = 0.0

if __name__ == "__main__":
    # a = Tensor(np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]]))
    # b = Tensor(np.array([[-1,-7,-3],[4,5,6]]))
    # bias = Tensor(np.ones_like(b.v@a.v)*2)

    # a = Tensor(np.array([[1, 2],
    #                      [3, 4],
    #                      [5, 6]]))
    # b = Tensor(np.array([[7, 8, 9],
    #                      [10, 11, 12],
    #                      [13, 14, 15]]))
    # bias = Tensor(np.ones_like(b.v @ a.v) * 2)


    # print("a", a)
    # print("b", b)
    # print("bias", bias)

    # f = b @ a + bias
    # # print("f", f)
    
    # f = f.relu()
    # print("f", f)

    # f.backward()

    # print("a grad", a.grad)
    # print("b grad", b.grad)
    # print("bias grad", bias.grad)


    a = Tensor(np.array([1],dtype=float), custom_name="a")
    b = Tensor(np.array([2],dtype=float), custom_name="b")
    c = Tensor(np.array([1],dtype=float), custom_name="c")
    d = Tensor(np.array([2],dtype=float), custom_name="d")
    e = Tensor(np.array([3, 3],dtype=float), custom_name="e")

    ab = a * b
    cd = c * d

    f = ab.cat([cd], 0)

    # f = Tensor(np.array([3, 3],dtype=float), custom_name="f")

    g = f @ e
    # h = g.T
    g.backward()
    print(ab)
    print(cd)
    print(f)
    print(e)
    print(g)

    #%%
    # import numpy as np
    # np.array([[3], [3]]) @  np.array([[3, 3]])
