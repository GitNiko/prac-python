# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import mnist_loader
#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):
  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.model = torch.nn.Sequential(
      torch.nn.Linear(sizes[0], sizes[1]),
      torch.nn.ReLU(),
      torch.nn.Linear(sizes[1], sizes[2]),
      torch.nn.ReLU(),
    )
  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
        random.shuffle(training_data)
        # 切割数组[1,2,3,4,5] => [[1,2], [3,4], [5]]
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta)
        if test_data:
            print("Epoch {0}: {1} / {2}".format(
                j, self.evaluate(test_data), n_test))
        else:
            print("Epoch {0} complete".format(j))

  def update_mini_batch(self, mini_batch, eta):
    # todos: 数组转换成tensor
    X = np.array([x for x, _ in mini_batch])
    Y = np.array([x for x, _ in mini_batch])
    x_tensor = torch.from_numpy(X)
    x = Variable(x_tensor)
    print(x.data.shape)
    pred = self.model(x)
    loss = torch.nn.MSELoss(size_average=False)(pred, Y)
    self.model.zero_grad()
    loss.backprop()
    for param in self.model.parameters():
      param.data -= eta * param.grad.data

  def evaluate(self, test_data):
    test_results = [(np.argmax(self.model(x)), y) for (x,y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
