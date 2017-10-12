import numpy as np
import cv2

im = cv2.imread('./images/j.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
# CHAIN_APPROX_SIMPLE 保存轮廓的方式，有全部保存点，或者保存一部分点,etc
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 参数: 原图片, 轮廓数组, 轮廓index, 轮廓颜色, 轮廓厚度
cv2.drawContours(im, contours, -1, (255,0,0), 1)
cv2.imshow('contours', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

