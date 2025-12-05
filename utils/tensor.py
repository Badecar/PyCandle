from typing import Sequence
import numpy as np
from utils import candle


class Tensor:
    """
    A tensor which holds a array and enables gradient computations.
    """

    def __init__(self, val: np.ndarray|list, grad_fn=lambda: [], custom_name=None, requires_grad:bool = True):
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
            orig_shape = grad_fn.get("orig_shape", None)
            permute_back = grad_fn.get("permute_back", None)
            im2_col_info = grad_fn.get("im2_col_info", None)
            max_pool_info = grad_fn.get("max_pool_info", None)

            if debug:
                print("custom_name", self.custom_name)
                print("bp", bp)
                print("grad", grad)

            if matrix_side != None:
                if matrix_side == "L":
                    input.backprop(bp @ grad)
                elif matrix_side == "R":
                    input.backprop(grad @ bp)

            elif permute_back is not None:
                input.backprop(bp.transpose(permute_back))

            elif orig_shape != None:
                bp = bp.reshape(orig_shape)
                input.backprop(bp)

            elif im2_col_info != None:
                bp = self.col2img(im2_col_info, bp)
                input.backprop(bp)

            elif max_pool_info is not None:
                bp = self.max_pool2d_bp(max_pool_info, bp)
                input.backprop(bp)
            
            elif grad_fn.get("T", None) != None:
                input.backprop(bp.T)
                
            # TODO: CAT PART OF BACKPROP NOT FIXED YET!!!
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

    def permute(self, *axes):
        # Calculate the new shape based on the permutation
        # axes will be a tuple like (1, 0, 2, 3)
        
        # Helper to define the backward pass (grad_fn)
        # The backward of a permute is just permuting back to the original order (argsort)
        def grad_fn():
            inverse_axes = np.argsort(axes)
            return [{
                "input": self, 
                "grad": None, 
                "permute_back": inverse_axes 
            }]

        new_v = self.v.transpose(axes)
    
        return Tensor(
            new_v, 
            grad_fn, 
            requires_grad=self.requires_grad
        )

    def img2col(self, stride:int, kernels, padding:int, kernel_size:tuple[int,int], in_channels:int):
        shape = self.shape #(Batch, Channel, Height, Width)
        N, C, H, W = shape
        KH, KW = kernel_size

        h_out = (H + 2 * padding - KH) // stride + 1
        w_out = (W + 2 * padding - KW) // stride + 1
        
        if padding > 0:
            v = np.pad(self.v, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
        else:
            v = self.v

        col = []
        for i in range(0, H + 2*padding - KH + 1, stride):
            for j in range(0, W + 2*padding - KW + 1, stride): #looping over all start positions of the kernel
                patch = v[:, :, i:i+KH, j:j+KW]
                # Flatten patch to (C * KH * KW)
                col.append(patch.reshape(N, -1)) #Using Python append and not np.append (O(1) vs O(N^2))

        col_stacked = np.array(col) #(h_out*w_out, N, C*KH*KW)
        col_stacked = col_stacked.transpose(1, 0, 2) #(N, h_out*w_out, C*KH*KW)
        col_matrix = col_stacked.reshape(-1, C * KH * KW) #(N * h_out*w_out, C*KH*KW)
        col_matrix = col_matrix.T #(C*KH*KW, N * h_out*w_out)

        def grad_fn():
            return [{
                "input": self,
                "grad": None,
                "im2_col_info": [padding, stride, shape, kernel_size]
            }]

        out_tensor = Tensor(
            col_matrix,
            grad_fn,
            requires_grad=self.requires_grad,
        )

        return out_tensor, (N, h_out, w_out)

    def col2img(self, im2_col_info, grad): # implemented as the opposite operation as in im2col
        padding, stride, shape, kernel_size = im2_col_info
        N, C, H, W = shape
        KH, KW = kernel_size

        h_out = (H + 2 * padding - KH) // stride + 1
        w_out = (W + 2 * padding - KW) // stride + 1

        grad = grad.reshape(C, KH, KW, N, h_out, w_out) #for easier looping
        grad = grad.transpose(3, 0, 4, 5, 1, 2) #(N, C, h_out, w_out, KH, KW) , Batch -> Channel -> slide over image -> kernel

        grad_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=grad.dtype)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                w_start = j * stride

                # Add the gradients from this kernel position back to the image block
                grad_patch = grad[:, :, i, j, :, :] # taking all gradients for all N and C at a specific kernel position for all entries in the KH and KW
                                                    # becomes (N, C, KH, KW)
                grad_padded[:, :, h_start:h_start+KH, w_start:w_start+KW] += grad_patch # Adding to a 2D "area" of the output grad, also of shape (N, C, KH, KW)

        if padding > 0:
            return grad_padded[:, :, padding:-padding, padding:-padding]
        return grad_padded


    def max_pool2d(self, kernel_size: tuple[int, int], stride: int = None, padding: int = 0): # Implementing this almost exactly as img2col. Just treating each C as an independent img as well as N (C*N in shape)
        if stride is None:
            stride = kernel_size[0]

        N, C, H, W = self.shape
        KH, KW = kernel_size

        x_reshaped = self.v.reshape(N * C, 1, H, W) #(N, C, H, W) -> (N*C, 1, H, W)
        

        if padding > 0:
            x_padded = np.pad(x_reshaped, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant', constant_values=-np.inf)
        else:
            x_padded = x_reshaped
            
        h_out = (H + 2 * padding - KH) // stride + 1
        w_out = (W + 2 * padding - KW) // stride + 1
        

        col = []
        for i in range(0, H + 2 * padding - KH + 1, stride):
            for j in range(0, W + 2 * padding - KW + 1, stride):
                patch = x_padded[:, :, i:i+KH, j:j+KW]
                col.append(patch.reshape(N * C, -1))
                
        # Stack and reshape to (KH*KW, (N*C)*h_out*w_out)
        col = np.array(col) # (h_out*w_out, N*C, KH*KW)
        col = col.transpose(1, 0, 2) # (N*C, h_out*w_out, KH*KW)
        col = col.reshape(-1, KH * KW).T # (KH*KW, N*C*h_out*w_out)
        
        out_vec = np.max(col, axis=0)
        max_indices = np.argmax(col, axis=0) #indices of the max values for backprop
        
        # 4. Reshape output back to (N, C, h_out, w_out)
        out_reshaped = out_vec.reshape(N, C, h_out, w_out)

        def grad_fn():
            return [{
                "input": self,
                "grad": None,
                "max_pool_info": {
                    "indices": max_indices,
                    "col_shape": col.shape,
                    "im2col_info": [padding, stride, (N * C, 1, H, W), kernel_size],
                    "orig_shape": (N, C, H, W)
                }
            }]

        return Tensor(out_reshaped, grad_fn, requires_grad=self.requires_grad)

    def max_pool2d_bp(self, max_pool_info, grad):
        indices = max_pool_info["indices"]
        im2col_params = max_pool_info["im2col_info"]
        col_shape = max_pool_info["col_shape"]
        orig_shape = max_pool_info["orig_shape"]
        
        grad_flat = grad.flatten()
        d_col = np.zeros(col_shape)
        
        # Route gradients to the max index for each column
        for i in range(len(indices)):
            row_idx = indices[i]
            d_col[row_idx, i] = grad_flat[i]
            
        d_im = self.col2img(im2col_params, d_col)
        
        return d_im.reshape(orig_shape)


    ## ACTIVATION FUNCTIONS ##
    def relu(self):
        return Tensor(np.maximum(self.v, 0.0),
        lambda: [{"input": self, "grad": (self.v > 0.0).astype(float)}],
        requires_grad=self.requires_grad)
    
    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.v))
        return Tensor(sig,
        lambda: [{"input": self, "grad": sig * (1 - sig)}],
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
