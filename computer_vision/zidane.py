import numpy as np
import cv2
from matplotlib import pyplot as plt

# Color image loaded by OpenCV is in BGR mode. But Matplotlib displays in RGB mode
# todo BGR => RGB

img = cv2.imread('./images/zidane.jpg')
# cv2.namedWindow('zidane', cv2.WINDOW_NORMAL)
# cv2.imshow('zidane', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('output.png', img)

# [[[b,g,r]]] => [[[r,g,b]]]
# 三种写法
########################################
# rgb = [[[point[2], point[1], point[0]] for point in lines] for lines in img]
##########################################
# for s in img:
#   for point in line:
#     y, x = len(line) - 1, len(point) - 1
#     t = img.item(x, y, 0)
#     img.itemset((x, y, 0), img.item(x, y, 2))
#     img.itemset((x, y, 2), t)

# print(img.dtype)
##########################################
t = np.copy(img[:, :, 0])
img[:, :, 0] = img[:, :, 2]
img[:, :, 2] = t
plt.imshow(img,  interpolation='bicubic')
# 坐标轴是否显示
# plt.xticks([])
# plt.yticks([])
plt.show()