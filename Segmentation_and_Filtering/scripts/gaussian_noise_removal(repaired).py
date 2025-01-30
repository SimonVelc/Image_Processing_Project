# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:34:15 2022

@author: saimo
"""

import cv2
import numpy as np

# 1ο
img = cv2.imread('rick.jpg')  ##Διάβασμα εικόνας και μετατροπή και grayscale

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.imshow('Original', img)

cv2.waitKey()

cv2.namedWindow('Grayscale', cv2.WINDOW_NORMAL)
cv2.imshow('Grayscale', gray)

cv2.waitKey()


# 3 adding Gaussian Noise
def Gaus(imgGray, mean=0, var=0.001):
    image = np.array(imgGray / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
    return out


gausImg = Gaus(gray, 0, 0.001)
cv2.namedWindow("Gaus image", cv2.WINDOW_NORMAL)
cv2.imshow("Gaus image", gausImg)
cv2.waitKey()


# 4 remove noise
def Gaus(imgGray, mean=0, var=0.001):
    image = np.array(imgGray / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


gausImg = Gaus(gray, 0, 0.001)
cv2.namedWindow("Gaus image", cv2.WINDOW_NORMAL)
cv2.imshow("Gaus image", gausImg)
cv2.imwrite('Gaus.jpg', gausImg)
cv2.waitKey()

# 5 apply Canny on the corrected grayscale
Corrected = cv2.Canny(gausImg, 20, 22, edges=None, apertureSize=3)
cv2.namedWindow("Repaired Img", cv2.WINDOW_NORMAL)
cv2.imshow("Repaired Img", Corrected)
cv2.imwrite('Repaired.jpg', Corrected)
cv2.waitKey()

cv2.destroyAllWindows()
