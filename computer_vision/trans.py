import cv2
import numpy as np

img = cv2.imread('./images/zidane.jpg', 0)
rows, cols = img.shape
##############################
# res = cv2.resize(img, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
#######################
# 𝚍𝚜𝚝(x,y)=𝚜𝚛𝚌(𝙼11x+𝙼12y+𝙼13,𝙼21x+𝙼22y+𝙼23)
# M = np.float32([[1, 0, 100], [0, 1, 50]])
# res = cv2.warpAffine(img, M, (cols, rows))
######################
# M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
# res = cv2.warpAffine(img, M, (cols, rows))
######################
# 原图三个点
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
# 转换后的图的三个点的位置
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
res = cv2.warpAffine(img, M, (cols, rows))

############################

# pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
# M = cv2.getPerspectiveTransform(pts1,pts2)
# res = cv2.warpPerspective(img,M,(300,300))


cv2.imshow('trans', res)
cv2.waitKey(0)
cv2.destroyAllWindows()