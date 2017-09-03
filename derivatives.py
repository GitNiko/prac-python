import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import newton

def f(x):
  return 3 * (x**2) + 2 * (x**5) + 3

def df(x):
  return 6 * x + 10 * (x**4)

def costFunc(resultSet, expected):
  return np.subtract(resultSet, expected)

def newtowns(f, df, x):
  return x - f(x)/df(x)

def dx(f, x):
    return abs(0-f(x))

def newtons_method(f, df, x0, e):
    delta = dx(f, x0)
    while delta > e:
        print("delta ", delta)
        x0 = x0 - f(x0)/df(x0)
        delta = dx(f, x0)
        print('Root is at: ', x0)
        print('f(x) at root is: ', f(x0))

newtons_method(f, df, -1, 0.00001)

print(newton(f, -1, df, tol=0.00001))

# a = 0
# x = -1.35

# while a < 15:
#   before = f(x)
#   x = s(f, df, x)
#   after = f(x)
#   print(x, after, abs(before) > abs(after))
#   a = a + 1




# t = np.arange(-20.0, 20.0, 0.1)
# print('training sets: {}'.format(t.size))
# print('cost result: {}'.format(costFunc(f(t), f(t))))



# plt.figure(1)
# plt.subplot(211)
# plt.plot(t, f(t))

# plt.subplot(212)
# plt.plot(t, df(t))

# plt.show()