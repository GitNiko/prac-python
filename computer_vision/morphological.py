import cv2
import numpy as np


img = cv2.imread('./images/j.png')
kernel = np.ones((5,5), np.uint8)
# print(kernel.shape)
# res = cv2.erode(img, kernel, iterations=1)

res = cv2.dilate(img, kernel, iterations=1)


cv2.imshow('erosion', res)
cv2.waitKey(0)
cv2.destroyAllWindows()