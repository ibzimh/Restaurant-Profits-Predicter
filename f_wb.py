import numpy as np

def fix_shape(X):
  # convert to 2d
  if (len(X.shape) == 1):
    X = np.array([X]).reshape(-1,1)
  return X

def f_wb_lin(X, w, b):
  return np.dot(fix_shape(X), w) + b