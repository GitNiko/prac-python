import cv2
import numpy as np 
from matplotlib import pyplot as plt

# img = cv2.imread('./images/zidane.jpg',0)
# plt.hist(img.ravel(), 256, [0, 256])
# plt.show()

# print(cv2.calcHist([img], [0], None, [256], [0, 256]))

img = cv2.imread('./images/zidane.jpg')
color = ('b', 'g', 'r')
for i, col in enumerate(color):
  histr = cv2.calcHist([img], [i], None, [256], [0, 256])
  plt.plot(histr, color=col)
  # plt.xlim([0, 256])

plt.show()