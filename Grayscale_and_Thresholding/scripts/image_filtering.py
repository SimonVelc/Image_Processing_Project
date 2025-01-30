# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 19:24:31 2022

@author: saimo
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('bart.jpg')
plt.imshow(img)
plt.show()


# 4 Segmetation

def segment_simp(img):
    ''' Attempts to segment the pixies out of the provided image '''

    #conversion to HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    cv2.namedWindow('hsv_image', cv2.WINDOW_NORMAL)
    cv2.imshow('hsv_image', hsv_image)
    cv2.imwrite('hsv_image.jpg', hsv_image)
    cv2.waitKey()

    #set blue range
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    #apply blue mask
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    #set white range
    light_white = (0, 0, 200)
    dark_white = (145, 60, 255)

    #apply the white mask
    mask_white = cv2.inRange(hsv_image, light_white, dark_white)

    #combine masks
    final_mask = mask + mask_white
    result = cv2.bitwise_and(img, img, mask=final_mask)

    #frame the segmentation using blur
    blur = cv2.GaussianBlur(result, (7, 7), 0)
    return blur

    plt.imshow(blur)
    plt.show()


result = segment_simp(img)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow('result', result)
cv2.imwrite('result.jpg', result)
cv2.waitKey()

cv2.destroyAllWindows()