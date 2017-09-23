import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./images/zidane.jpg')
# kernel = np.ones((5,5), np.float32)/250
# # convolve 卷积
# dst = cv2.filter2D(img, -1, kernel)

# dst = cv2.blur(img,(10,10))
dst = cv2.GaussianBlur(img,(5,5),20)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.show()
