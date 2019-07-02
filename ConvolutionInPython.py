# -*- coding: utf-8 -*-
"""
Created on Sat May 18 22:00:32 2019

@author: Ujjawal
"""

import numpy as np
import os
import matplotlib.pyplot as plt

import cv2


img_dog = cv2.imread(r'F:\CourseraTensorflow\dog.png')
cv_grey = cv2.cvtColor(img_dog,cv2.COLOR_BGR2GRAY)
cv_greyCopy = np.copy(cv_grey)
plt.imshow(cv_grey,cmap='gray')

filter = [[1,0,-1],
          [2,0,-2],
          [1,0,-1]]

size_x= cv_grey.shape[0]
size_y=cv_grey.shape[1]

#Convolution
for i in range(1,size_x-1):
    for j in range(1,size_y-1):
        conv=0.0
        conv = conv+(cv_grey[i-1,j-1] * filter[0][0])
        conv = conv+(cv_grey[i,j-1] * filter[1][0])
        conv = conv+(cv_grey[i+1,j-1] * filter[2][0])
        conv = conv+(cv_grey[i-1,j] * filter[0][1])
        conv = conv+(cv_grey[i+1,j] * filter[2][1])
        conv = conv+(cv_grey[i-1,j+1] * filter[0][2])
        conv = conv+(cv_grey[i,j+1] * filter[1][2])
        conv = conv+(cv_grey[i+1,j+1] * filter[2][2])
        
        if(conv<0):
            conv=0
        if(conv>255):
            conv = 255
        cv_greyCopy[i,j]= conv


#SecondConvolution
 

        

#Polling

pooled_x=int (size_x/16)
polled_y =int (size_y/16)

pooled_image = np.zeros((pooled_x,polled_y))
print(pooled_image.shape)
print(cv_greyCopy.shape)
  
for x in range(0,size_x-16,16):
    for y in range(0,size_y-16,16):
        pixels = []
        pixels.append(cv_greyCopy[x,y])
        pixels.append(cv_greyCopy[x+1,y])
        pixels.append(cv_greyCopy[x,y+1])
        pixels.append(cv_greyCopy[x+1,y+1])
        pooled_image[int(x/16),int(y/16)] = max(pixels)
        
plt.imshow(pooled_image,cmap='gray')
