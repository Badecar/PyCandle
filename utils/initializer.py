import random
import numpy as np
from utils.cn import Parameter
from abc import ABC, abstractmethod
class Initializer(ABC):
  @abstractmethod
  def init_weights(self, n_in, n_out):
    pass
class NormalInitializer(Initializer):
  def __init__(self, mean=0, std=None):
    self.mean = mean
    self.std = std  # If None, will use Xavier initialization

  def init_weights(self, n_in, n_out):
    # Use Xavier initialization if std not provided
    if self.std is None:
      std = np.sqrt(2.0 / (n_in + n_out))  # Xavier/Glorot initialization
    else:
      std = self.std
    return Parameter(np.random.normal(self.mean, std, size=(n_in,n_out)))

  def init_bias(self, n_out):
    return Parameter(np.zeros([n_out]))

class ConstantInitializer(Initializer):
  def __init__(self, weight=1.0, bias=0.0):
    self.weight = weight
    self.bias = bias

  def init_weights(self, n_in, n_out):
    return Parameter(np.ones([n_in,n_out]) * self.weight)

class UniformInitializer(Initializer):
    def __init__(self):
      pass

    def init_weights(self, n_in, n_out):
      k = 1/np.sqrt(n_in)
      return Parameter(np.random.uniform(-k, k, size=(n_in, n_out)))

    def init_bias(self, n_out):
      k = 1/np.sqrt(n_out)  
      return Parameter(np.random.uniform(-k, k, size=(n_out)))