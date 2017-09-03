import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return np.square(x)

def dfunc(x):
  return 2 * x

def GD(x_start, df, epochs, lr):
    xs = np.zeros(epochs + 1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
      dx = df(x)
      v = -dx * lr
      x += v
      xs[i+1] = x
    return xs

def demo0_GD():
  line_x = np.linspace(-5, 5, 100)
  line_y = func(line_x)
  
  x_start = -5
  epochs = 5
  lr = 0.3
  x = GD(x_start, dfunc, epochs, lr=lr)

  fig = plt.figure(figsize=(6,6))
  color = 'r'
  plt.plot(line_x, line_y, c='b')
  plt.plot(x, func(x), c=color)
  plt.scatter(x, func(x), c=color)
  # plt.legend()
  plt.show()

demo0_GD()