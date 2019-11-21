# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:45:00 2019

@author: Mikko
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:29:14 2019

@author: Mikko
"""

 # Q4 # 
# Training code:
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten,Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import sklearn.model_selection



# First we initialize the model. "Sequential" means there are no loops.
model = Sequential()
   
N = 10 # Number of feature maps
w, h = 5, 5 # Conv. window size

model.add(Conv2D(N, (w, h),input_shape=(64, 64, 3),activation = "relu",padding = "same"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(N, (w, h),activation = "relu",padding = "same"))

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(2, activation = "sigmoid"))
model.summary()


