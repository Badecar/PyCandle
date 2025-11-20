import random
import numpy as np
from tensor import Tensor

class Initializer:

  def init_weights(self, n_in, n_out):
    raise NotImplementedError

  def init_bias(self, n_out):
    raise NotImplementedError


class NormalInitializer(Initializer):

  def __init__(self, mean=0, std=0.1):
    self.mean = mean
    self.std = std

  def init_weights(self, n_in, n_out):
    return Tensor(np.random.normal(self.mean, self.std, size=(n_in,n_out)))

  def init_bias(self, n_out, batch_size):
    return Tensor(np.zeros([n_out, batch_size]))

class ConstantInitializer(Initializer):

  def __init__(self, weight=1.0, bias=0.0):
    self.weight = weight
    self.bias = bias

  def init_weights(self, n_in, n_out):
    return Tensor(np.ones([n_in,n_out]) * self.weight)

  def init_bias(self, n_out, batch_size):
    return Tensor(np.ones([n_out, batch_size]) * self.bias)