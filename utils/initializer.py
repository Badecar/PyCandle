import random
import numpy as np
from utils.var import Var

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
    return np.array([[Var(random.gauss(self.mean, self.std)) for _ in range(n_out)] for _ in range(n_in)])

  def init_bias(self, n_out):
    return np.array([Var(0.0) for _ in range(n_out)])

class ConstantInitializer(Initializer):

  def __init__(self, weight=1.0, bias=0.0):
    self.weight = weight
    self.bias = bias

  def init_weights(self, n_in, n_out):
    return np.array([[Var(self.weight) for _ in range(n_out)] for _ in range(n_in)])

  def init_bias(self, n_out):
    return np.array([Var(self.bias) for _ in range(n_out)])