import numpy as np
import math
from f_wb import fix_shape

def gradient_descent(X, y, w, b, cost_function, gradient_function, alpha, num_iters):
  X = fix_shape(X)
  J_history = []
  wb_history = []

  m, n = X.shape # number of training examples, 

  for i in range(num_iters):
    dj_dw, dj_db = gradient_function(X, y, w, b)
    w = w - alpha * dj_dw
    b = b - alpha * dj_db
    
    J_history.append(cost_function(X, y, w, b))
    wb_history.append([w, b])

    # Save cost J at each iteration
    if i < 100000:      # prevent resource exhaustion 
        cost =  cost_function(X, y, w, b)
        J_history.append(cost)

    # Print cost every at intervals 10 times or as many iterations if < 10
    if i % math.ceil(num_iters/10) == 0:
        wb_history.append(w)
        print(f"Iteration {i:4}: Cost {float(J_history[-1])}")

  return w, b, wb_history, J_history