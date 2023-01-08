import numpy as np
import matplotlib.pyplot as plt

def load_data():
  data = np.loadtxt('ex1data1.txt', delimiter=',')
  X = data[:,0]
  y = data[:,1]
  return X, y

def print_data_info(x, y):
  print(f'The type of x_train is : {type(x)}')
  print(f'The first five elements of x_train are : {x[:5]}')
  print(f'The type of y_train is : {type(y)}')
  print(f'The first five elements of y_train are : {y[:5]}')
  print(f'The shape of x_train is {x.shape}')
  print(f'The shape of y_train is {y.shape}')

def plot_input_data(x, y):
  plt.scatter(x, y, c='r', marker='x', label='data')
  plt.title('Population vs Profits')
  plt.xlabel('Population in 10,000\'s')
  plt.ylabel('Profits in $10,000\'s')
  plt.legend()
  plt.show()

def plot_predicted_data(x, predicted, y):
  plt.plot(x, predicted, c = "b", label='predicted')
  plt.scatter(x, y, marker='x', c='r', label='data') 

  plt.title("Profits vs. Population per city")
  plt.ylabel('Profit in $10,000')
  plt.xlabel('Population of City in 10,000s')
  plt.legend()
  plt.show()

