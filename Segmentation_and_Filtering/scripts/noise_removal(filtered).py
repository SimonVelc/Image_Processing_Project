# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 21:24:53 2022

@author: saimo
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

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
    return out


gausImg = Gaus(gray, 0, 0.001)
cv2.namedWindow("Gaus image", cv2.WINDOW_NORMAL)
cv2.imshow("Gaus image", gausImg)
cv2.imwrite('Gaus.jpg', gausImg)
cv2.waitKey()


# 4 remove noise
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err


imageGaus = cv2.GaussianBlur(gausImg, (5, 5), 1)
cv2.namedWindow('imageGaus', cv2.WINDOW_NORMAL)
cv2.imshow('imageGaus', imageGaus)
cv2.waitKey()

# Mean square error
print("***************************************")
print("Mean square error for Gaussian", mse(gausImg, imageGaus))

# SSIM score
simil_score, _ = ssim(gausImg, imageGaus, full=True)
print('SSIM score for Gaussian is:{:.3f}'.format(simil_score))
cv2.imwrite('filtered.jpg', imageGaus)
cv2.waitKey()

cv2.destroyAllWindows()