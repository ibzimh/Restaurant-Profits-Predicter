import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from utils import *
from cost_function import compute_cost
from f_wb import f_wb_lin
from gradient_function import compute_gradient
from gradient_descent import gradient_descent

x_train, y_train = load_data() # population in 10,000's, avg. montly profits in $10,000's

# print_data_info(x_train, y_train)
# plot_input_data(x_train, y_train)

from public_tests import *
# compute_cost_test(compute_cost)
# compute_gradient_test(compute_gradient)

# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)

predicted = f_wb_lin(x_train, w, b)

predict1 = 3.5 * w[0] + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))

predict2 = 7.0 * w[0] + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))

plot_predicted_data(x_train, predicted, y_train)