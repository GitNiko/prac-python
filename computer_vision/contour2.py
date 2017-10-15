import cv2
import numpy as np

orgImg = cv2.imread('./images/polygon.jpg')
img = cv2.cvtColor(orgImg, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 127, 255, 0)
thresh = cv2.bitwise_not(thresh)
im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv2.moments(cnt)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

epsilon = cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
hull = cv2.convexHull(cnt)
x,y,w,h = cv2.boundingRect(cnt)

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)

## calc the contour properties
area = cv2.contourArea(cnt)
hull_area = cv2.contourArea(hull)
print('aspect ratio: {}'.format(float(w)/h))
print('extend: {}'.format(float(area)/w * h))
print('solidity: {}'.format(float(area)/hull_area))
print('equivalent: {}'.format(np.sqrt(4 * area / np.pi)))
print('orientation: {}'.format(cv2.fitEllipse(cnt)))

cv2.rectangle(orgImg, (x, y), (x + w, y + h), (0, 255, 0), 1)
cv2.drawContours(orgImg, [box], -1, (0,255,0), 1)
cv2.imshow('contours', orgImg)
cv2.waitKey(0)
cv2.destroyAllWindows()