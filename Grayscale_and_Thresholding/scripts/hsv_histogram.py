# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 19:05:51 2022

@author: saimo
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


#1 reading image and 3D plotting
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(len(flags))
print(flags[40])


simp = cv2.imread('bart.jpg')
plt.imshow(simp)
plt.show()

r, g, b = cv2.split(simp)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = simp.reshape((np.shape(simp)[0]*np.shape(simp)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()

#2 HSV and 3D plot
hsv_simp = cv2.cvtColor(simp, cv2.COLOR_RGB2HSV)

h, s, v = cv2.split(hsv_simp)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()
plt.figure(0)
plt.subplot(2,2,1)

plt.imshow(simp)

