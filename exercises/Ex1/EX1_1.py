# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt


# A)
from scipy.io import loadmat
mat = loadmat("twoClassData.mat")

print(mat.keys())
X = mat["X"]
y = mat["y"].ravel()
 
# B)
#plt.plot(X[y==0,:],'blue')
plt.plot(X[y == 0,0],X[y == 0,1],'ro')
plt.plot(X[y == 1,0],X[y == 1,1],'bo')
