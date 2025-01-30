# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 21:14:36 2022

@author: saimo
"""
import cv2

# 1 image reading and conversion to Grayscale
img = cv2.imread('rick.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.imshow('Original', img)

cv2.waitKey()

cv2.namedWindow('Grayscale', cv2.WINDOW_NORMAL)
cv2.imshow('Grayscale', gray)
cv2.imwrite('Grayscale.jpg', gray)

cv2.waitKey()

# 2 canny technique for edge detection
Canny = cv2.Canny(img, 20, 22, edges=None, apertureSize=3)
cv2.namedWindow("Canny img", cv2.WINDOW_NORMAL)
cv2.imshow("Canny img", Canny)
cv2.imwrite('Canny.jpg', Canny)
cv2.waitKey()

cv2.destroyAllWindows()