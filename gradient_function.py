import numpy as np
from f_wb import f_wb_lin, fix_shape

def compute_gradient(X, y, w, b):
  X = fix_shape(X)
  f_wb = f_wb_lin(X, w, b) # an array of (+ ... w_i * x_i + ... + b)'s
  m, n = X.shape # no. of training examples, 

  dj_dw = 0
  dj_db = 0
  for i in range(m):
    dj_dw += (f_wb[i] - y[i])* X[i]
    dj_db += (f_wb[i] - y[i])

  return (dj_dw / m), (dj_db / m)