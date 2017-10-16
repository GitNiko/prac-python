import cv2
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('./images/polygon.jpg',0)
# plt.hist(img.ravel(), 256, [0, 256])
# plt.show()

print(cv2.calcHist([img], [0], None, [256], [0, 256]))
