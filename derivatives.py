import numpy as np 
import matplotlib.pyplot as plt

def f(x):
  return 3 * (x**2) + 2 * (x**5)

def df(x):
  return 6 * x + 10 * (x**4)

t = np.arange(-20.0, 20.0, 0.1)

plt.figure(1)
plt.subplot(211)
plt.plot(t, f(t))

plt.subplot(212)
plt.plot(t, df(t))

plt.show()