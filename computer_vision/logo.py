import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('./images/zidane.jpg')
img2 = cv2.imread('./images/logo.png')
# rows, cols, channels = img2.shape
# img1 = img1[0:rows, 0:cols]

# img1 = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

# cv2.imshow('dst', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

rows, cols, channels = img2.shape
roi = img1[0: rows, 0: cols]
# 灰度化
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# 阀值操作()
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# (roi & roi) && (mask == 0)
# 提取出roi区域的值
img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
# 去掉roi区域的值
img2_fg = cv2.bitwise_and(img2, img2, mask = mask)

# 合成区域值
dst = cv2.add(img1_bg, img2_fg)
img1[0: rows, 0: cols] = dst

cv2.imshow('res', img2_fg)
cv2.waitKey(0)
cv2.destroyAllWindows()