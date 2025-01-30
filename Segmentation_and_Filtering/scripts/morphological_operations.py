# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:50:18 2022

@author: saimo
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 1 #image reading and conversion to Grayscale
img = cv2.imread('rick.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

cv2.imshow('Original', img)

cv2.waitKey()

cv2.namedWindow('Grayscale', cv2.WINDOW_NORMAL)
cv2.imshow('Grayscale', gray)

cv2.waitKey()

# 2 canny to detect edges
Canny = cv2.Canny(img, 20, 22, edges=None, apertureSize=3)
cv2.namedWindow("Canny img", cv2.WINDOW_NORMAL)
cv2.imshow("Canny img", Canny)

cv2.waitKey()


# 3 adding gaus noise
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
cv2.waitKey()


# 4 noise removal
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

# 5 apply Canny on the corrected grayscale
Corrected = cv2.Canny(gausImg, 20, 22, edges=None, apertureSize=3)
cv2.namedWindow("Repaired Img", cv2.WINDOW_NORMAL)
cv2.imshow("Repaired Img", Corrected)
cv2.waitKey()

# 6 compare canny and canny from corrected grayscale
simil_score, _ = ssim(Corrected, Canny, full=True)
print('SSIM score for those two images is:{:.3f}'.format(simil_score))
cv2.waitKey()

# 7
kernel = np.ones((3, 3), np.uint8)

# προσέγγιση dilation
Dilate = cv2.dilate(gausImg, kernel=kernel, iterations=1)
cv2.namedWindow("Dilate image", cv2.WINDOW_NORMAL)
cv2.imshow("Dilate image", Dilate)
cv2.imwrite('Dilate.jpg', Dilate)
cv2.waitKey()

# erosion approach
Erode = cv2.erode(gausImg, kernel=kernel, iterations=1)
cv2.namedWindow("Erode image", cv2.WINDOW_NORMAL)
cv2.imshow("Erode image", Erode)
cv2.imwrite('Erode.jpg', Erode)
cv2.waitKey()

# Technique Canny
CannyDilate = cv2.Canny(Dilate, 20, 22, edges=None, apertureSize=3)
cv2.namedWindow("Canny&Dilate image", cv2.WINDOW_NORMAL)
cv2.imshow("Canny&Dilate image", CannyDilate)
cv2.imwrite('Canny&Dilate.jpg', CannyDilate)
cv2.waitKey()

CannyErode = cv2.Canny(Erode, 20, 22, edges=None, apertureSize=3)
cv2.namedWindow("Canny&Erode image", cv2.WINDOW_NORMAL)
cv2.imshow("Canny&Erode image", CannyErode)
cv2.imwrite('Canny&Erode.jpg', CannyErode)
cv2.waitKey()

simil_score, _ = ssim(CannyDilate, Canny, full=True)
print('SSIM score for the images "Canny&Dilate" and "Canny" is:{:.3f}'.format(simil_score))
cv2.waitKey()

simil_score, _ = ssim(CannyErode, Canny, full=True)
print('SSIM score for the images "Canny&Erode" and "Canny" is:{:.3f}'.format(simil_score))
cv2.waitKey()

cv2.destroyAllWindows()