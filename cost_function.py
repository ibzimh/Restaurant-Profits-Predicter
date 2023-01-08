import numpy as np
from f_wb import fix_shape, f_wb_lin

def compute_cost(X, y, w, b):
  # convert to 2d iff 1d so the function works with single inputs too
  X = fix_shape(X)
  m, n = X.shape # no. of training examples, 
  f_wb = f_wb_lin(X, w, b)
  cost = 0
  
  for i in range(m):
    cost += (f_wb[i] - y[i]) ** 2
  
  return cost / (2 * m)