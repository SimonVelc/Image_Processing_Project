# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 21:07:56 2022

@author: saimo
"""

import cv2

#1 Διάβασμα εικόνας και μετατροπή σε Grayscale
img = cv2.imread('rick.jpg') 
	
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.imshow('Original', img) 

cv2.waitKey()

cv2.namedWindow('Grayscale', cv2.WINDOW_NORMAL)
cv2.imshow('Grayscale', gray) 
cv2.imwrite('Grayscale.jpg', gray)

cv2.waitKey()

cv2.destroyAllWindows()