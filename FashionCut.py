#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 22:33:20 2022

@author: dev
"""

from skimage import io
from skimage.transform import rescale
from skimage.filters import gaussian, sobel

import cv2
import numpy as np

import matplotlib.pyplot as plt



#cv2.imshow(img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print(r.shape)


img = cv2.imread("/Users/dev/Downloads/shirt.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

edge_img = cv2.Canny(img,300,600)

#print(edge_img.shape)
#img[400:480,:,0] = img[400:480,:,0]- 20*np.ones([80,500])




for row in range(230,700):
    s=[]
    start=0
    end=0
    
    for col in range(500 ):
        if edge_img[row][col]>150:
            s.append(col)
            
    start=s[0]
    end=s[len(s)-1]
    
    img[row,start:end,0] = 255*np.ones([1,end-start])
    #img[row,start:end,1] = img[row,start:end,0]-40*np.ones([1,end-start])
    #img[row,start:end,2] = img[row,start:end,0]+20*np.ones([1,end-start])


# Display
plt.subplot(221)
plt.imshow(img)

plt.subplot(222)
plt.imshow(edge_img)

plt.subplot(223)
plt.imshow(img[100:250,70:400,:])


