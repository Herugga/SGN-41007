# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:19:57 2019

@author: Oma
"""

import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np

# Read the data

img = imread("uneven_illumination.jpg")
plt.imshow(img, cmap='gray')
plt.title("Image shape is %dx%d" % (img.shape[1], img.shape[0]))
plt.show()

# Create the X-Y coordinate pairs in a matrix
X, Y = np.meshgrid(range(1300), range(1030))
Z = img

x = X.ravel()
y = Y.ravel()
z = Z.ravel()

# ********* TODO 1 **********
# Create data matrix
# Use function "np.column_stack".
# Function "np.ones_like" creates a vector like the input.

#h = np.column_stack((x,y,z))
#H = np.ones_like(h)
H = np.column_stack((x*x,y*y,x*y,x,y,np.ones_like(x)))


# ********* TODO 2 **********
# Solve coefficients
# Use np.linalg.lstsq
# Put coefficients to variable "theta" which we use below.
#theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
theta = np.linalg.lstsq(H,z)


# Predict
z_pred = H @ theta[0]
#z_pred = np.dot(H, theta)
Z_pred = np.reshape(z_pred, X.shape)

# Subtract & show
S = Z - Z_pred
plt.imshow(S, cmap = 'gray')
plt.show()