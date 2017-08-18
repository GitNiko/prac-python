import numpy as np 
import matplotlib.pylab as plt

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

# t = np.arange(0., 5., 0.2)

# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()
x = np.arange(-10., 10., 0.1)
line, = plt.plot(x, sigmoid(x), '-')
# line.set_antialiased(False)
plt.show()

