from typing import Sequence
import numpy as np
from utils import candle


class Tensor:
    """
    A tensor which holds a array and enables gradient computations.
    """

    def __init__(self, val: np.ndarray|list, grad_fn=lambda: [], custom_name=None, requires_grad:Bool = True):
        self.v = np.array(val, dtype=float)
        self.grad_fn = grad_fn
        self._grad = None
        self.custom_name = custom_name
        self.requires_grad = requires_grad and candle.is_grad_enabled()

        if not self.requires_grad:
            self.grad_fn = lambda: []

        if type(self.v) != np.ndarray:
            self.v = np.array(self.v)

    @property
    def T(self):
        return Tensor(self.v.T, lambda: [{"input": self, "grad": None, "T": "yes"}], requires_grad=self.requires_grad)
    
    @property
    def grad(self):
        return self._grad

    @property
    def shape(self):
        return self.v.shape

    def backprop(self, bp, debug=False):
        if self._grad is None:
            self._grad = np.zeros_like(self.v)

        # Handle broadcasting: if bp shape doesn't match self.v shape, sum to reduce
        bp = np.asarray(bp)
        if bp.shape != self.shape:
            # Sum over extra leading dimensions
            ndim_diff = len(bp.shape) - len(self.shape)
            for _ in range(ndim_diff):
                bp = bp.sum(axis=0)
            
            # Sum over dimensions that were broadcast (size 1 -> size N)
            for i in range(len(self.shape)):
                if self.shape[i] == 1 and bp.shape[i] > 1:
                    bp = bp.sum(axis=i, keepdims=True)

        self._grad += bp
        for grad_fn in self.grad_fn():
            input: Tensor = grad_fn["input"]
            grad: np.ndarray = grad_fn["grad"]

            start_index = grad_fn.get("start_index", None)
            end_index = grad_fn.get("end_index", None)
            matrix_side = grad_fn.get("matrix", None)
            transposed = grad_fn.get("T", None)
            orig_shape = grad_fn.get("orig_shape", None)

            if debug:
                print("custom_name", self.custom_name)
                print("bp", bp)
                print("grad", grad)

            if matrix_side != None:
                if matrix_side == "L":
                    input.backprop(bp @ grad)
                elif matrix_side == "R":
                    input.backprop(grad @ bp)
            
            elif transposed != None:
                input.backprop(bp.T)

            elif orig_shape != None:
                bp = bp.reshape(orig_shape)
                input.backprop(bp)
                
            # CAT PART OF BACKPROP NOT FIXED YET!!!
            elif start_index is not None and end_index is not None:
                input.backprop(grad @ bp[start_index:end_index])
            else:
                # Ensure bp and grad are arrays for proper broadcasting
                bp_array = np.asarray(bp)
                grad_array = np.asarray(grad)
                # For sum operation: if bp is scalar and grad is array, broadcast bp to grad's shape
                result = grad_array * bp_array
                input.backprop(result)

    def backward(self, debug=False):
        # self.backprop(np.array([1]))
        self.backprop(np.ones_like(self.v), debug=debug)
    
    def zero_grad(self,grad_none=False):
        if grad_none:
            self._grad = None
        else:
            self._grad = 0.0

    def cat(self, others: Sequence['Tensor'], axis: int):
        all = [self] + others
        current_index = 0
        outputs = []
        for part in all:
            output = {
                "input": part,
                "grad": np.ones_like(part.v),
                "start_index": current_index,
                "end_index": current_index + part.shape[axis]
            }
            outputs.append(output)
            current_index += part.shape[axis]
        requires_grad = all([o.requires_grad for o in all])
        return Tensor(
            np.concatenate([o.v for o in all], axis=axis),
            lambda: outputs,
            custom_name=f"cat({', '.join([o.custom_name for o in all])})",
            requires_grad=requires_grad
        )

    def flatten(self, start_dim: int = 0, end_dim: int = -1):
        shape = self.shape

        if start_dim < 0:
            start_dim = len(shape) + start_dim 
        if end_dim < 0:
            end_dim = len(shape) + end_dim

        if end_dim < start_dim:
            raise ValueError(f"end_dim of {end_dim} must be bigger than start_dim of {start_dim}")
        
        # Compose new shape
        start_shape = shape[:start_dim]
        flatten_dim = np.prod(shape[start_dim:end_dim+1])
        end_shape = shape[end_dim+1:]

        final_shape = start_shape + (flatten_dim,) + end_shape

        def grad_fn():
            return [{"input": self, "grad": None, "orig_shape": self.shape}]

        flat = self.v.reshape(final_shape)

        return Tensor(flat, grad_fn, requires_grad=self.requires_grad)
    
    def unflatten(self, unflatten_dim: int = 0, shape: tuple[int, ...] = ()):
        orig_shape = self.shape
        
        if unflatten_dim < 0:
            unflatten_dim = len(orig_shape) + unflatten_dim
        
        start_shape = orig_shape[:unflatten_dim]
        mid_shape = shape
        end_shape = orig_shape[unflatten_dim+1:]

        if orig_shape[unflatten_dim] != np.prod(shape):
            raise ValueError(f"Cannot unflatten: dimension size at unflatten_dim {unflatten_dim} is {orig_shape[unflatten_dim]}, but shape to unflatten {shape} has product {np.prod(shape)}.")

        final_shape = start_shape + mid_shape + end_shape

        def grad_fn():
            return [{"input": self, "grad": None, "orig_shape": self.shape}]
        
        unflat = self.v.reshape(final_shape)

        return Tensor(unflat, grad_fn, requires_grad=self.requires_grad)

    # def __getitem__(self, key): #NOTE: Log how many times this has been called and make sure backprop runs that many times before continuing
    #     return Tensor(self.v[key], )

    def __add__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        return Tensor(self.v + other.v, lambda: [{"input": self, "grad": np.ones_like(self.v)}, {"input": other, "grad": np.ones_like(other.v)}], custom_name=f"({self.custom_name} + {other.custom_name})", requires_grad=self.requires_grad or other.requires_grad)

    def __mul__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        return Tensor(self.v * other.v,
        lambda: [{"input": self, "grad": other.v}, {"input": other, "grad": self.v}],
        custom_name=f"({self.custom_name} * {other.custom_name})",
        requires_grad=self.requires_grad or other.requires_grad)

    def __matmul__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        if len(self.shape) == 1 and len(other.shape) == 1:
            return self.__mul__(other)
        if self.shape[-1] != other.shape[-2]:
            raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")
        return Tensor(self.v @ other.v,
        lambda: [{"input": self, "grad": other.v.T, "matrix": "L"}, {"input": other, "grad": self.v.T, "matrix": "R"}],
        custom_name=f"({self.custom_name} @ {other.custom_name})",
        requires_grad=self.requires_grad or other.requires_grad)

    def __pow__(self, power):
        assert type(power) in {float, int}, "power must be float or int"
        return Tensor(self.v ** power,
        lambda: [{"input": self, "grad": power * self.v ** (power - 1)}],
        custom_name=f"({self.custom_name} ** {power})",
        requires_grad=self.requires_grad)

    def __neg__(self: 'Tensor') -> 'Tensor':
        return Tensor(-self.v,
        lambda: [{"input": self, "grad": -np.ones_like(self.v)}],
        custom_name=f"-({self.custom_name})",
        requires_grad=self.requires_grad)

    def __sub__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        return Tensor(self.v - other.v, lambda: [
            {"input": self, "grad": np.ones_like(self.v)}, 
            {"input": other, "grad": -np.ones_like(other.v)}
        ],
        custom_name=f"({self.custom_name} - {other.custom_name})",
        requires_grad=self.requires_grad or other.requires_grad)

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
        return Tensor(result, lambda: [{"input": self, "grad": np.ones_like(self.v)}], requires_grad=self.requires_grad)
    
    def exp(self):
        ev = np.exp(self.v)
        return Tensor(ev, lambda: [{"input" : self, "grad" : ev}], requires_grad=self.requires_grad)

    def log(self):
        return Tensor(np.log(self.v), lambda: [{"input" : self, "grad" : self.v ** -1}], requires_grad=self.requires_grad)

    def img2col(self, stride:int, kernels, padding:int, kernel_size:tuple[int,int], in_channels:int):
        im_flat = np.zeros([kernels.v.size, self.v.size])
        h_out = int((self.v.shape[2] + 2 * padding - kernel_size[0]) // stride + 1)
        w_out = int((self.v.shape[3] + 2 * padding - kernel_size[1]) // stride + 1)
        batches = int(self.v.shape[0])
        channels = in_channels


        start_positions = 


        for i in range(kernels.v.size * channels): # Looping over positions in a kernel
            for j in range(h_out * w_out * batches): # Looping over all out_pixels
                img_nr = j // batches
                ch = i // (h_out * w_out)
                axis_0 = i // (kernel_size[0] - ch*kernel_size[0]) + j // (kernel_size[1] - img_nr*kernel_size[1])
                axis_1 = i % kernel_size[1] + j % kernel_size[1]
                im_flat[i,j] = self.v[img_nr, ch, axis_0, axis_1]

        im_flat = Tensor(im_flat)
        return im_flat


    ## ACTIVATION FUNCTIONS ##
    def relu(self):
        return Tensor(np.maximum(self.v, 0.0),
        lambda: [{"input": self, "grad": (self.v > 0.0).astype(float)}],
        requires_grad=self.requires_grad)
    


if __name__ == "__main__":
    a = Tensor(np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]]))
    print(a.shape)
    a = a.flatten()
    print(a.shape)
    a = a.unflatten(0, (3,4))
    print(a.shape)
    a = a.T
    a = a.T
    b = Tensor(np.array([[-1,-7,-3],[4,5,6]]))
    bias = Tensor(np.ones_like(b.v@a.v)*2)

    print("a", a)
    print("b", b)
    print("bias", bias)

    f = b @ a + bias
    # print("f", f)
    
    f = f
    print("f", f)

    f.backward(debug=False)
    

    print("a grad", a.grad)
    print("b grad", b.grad)
    print("bias grad", bias.grad)


    # a = Tensor(np.array([1],dtype=float), custom_name="a")
    # b = Tensor(np.array([2],dtype=float), custom_name="b")
    # c = Tensor(np.array([1],dtype=float), custom_name="c")
    # d = Tensor(np.array([2],dtype=float), custom_name="d")
    # e = Tensor(np.array([3, 3],dtype=float), custom_name="e")

    # ab = a * b
    # cd = c * d

    # f = ab.cat([cd], 0)

    # # f = Tensor(np.array([3, 3],dtype=float), custom_name="f")

    # g = f @ e
    # # h = g.T
    # g.backward()
    # print(ab)
    # print(cd)
    # print(f)
    # print(e)
    # print(g)

    #%%
    # import numpy as np
    # np.array([[3], [3]]) @  np.array([[3, 3]])
