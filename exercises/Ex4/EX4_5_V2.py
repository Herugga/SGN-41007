# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:29:14 2019

@author: Mikko
"""

 # Q5 #
 
 # Training code
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from simplelbp import local_binary_pattern
from sklearn.model_selection import cross_val_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_data(folder):
    """ 
    Load all images from subdirectories of
    'folder'. The subdirectory name indicates
    the class.
    """
    
    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here
    
    subdirectories = glob.glob(folder + "/*")
    
    # Loop over all folders
    for d in subdirectories:
        
        # Find all files from this folder
        files = glob.glob(d + os.sep + "*.jpg")
        
        # Load all files
        for name in files:
            
            # Load image and parse class name
            img = plt.imread(name)
            class_name = name.split(os.sep)[-2]

            # Convert class names to integer indices:
            if class_name not in classes:
                classes.append(class_name)
            
            class_idx = classes.index(class_name)
            
            X.append(img)
            y.append(class_idx)
    
    # Convert python lists to contiguous numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y
X, y = load_data(".")
# Split to training and evaluation:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train = X_train[..., np.newaxis] / 255.0
X_test  = X_test[..., np.newaxis] / 255.0

# Output has to be one-hot-encoded
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten,Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

model = Sequential()
   
N = 10 # Number of feature maps
w, h = 5, 5 # Conv. window size

model.add(Conv2D(32, (w, h),input_shape=(64, 64, 1),activation = "relu",padding = "same"))

model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(32, (w, h),activation = "relu",padding = "same"))

model.add(MaxPooling2D((4,4)))

model.add(Flatten())

model.add(Dense(100, activation = "sigmoid"))
model.add(Dense(2, activation = "sigmoid"))
model.summary()


 # Q5 #
model.compile(optimizer='sgd',loss='logcosh',metrics=['accuracy'])
model.fit(X_train,y_train, epochs=20, batch_size=32,validation_data = (X_test, y_test))    

 #Check accuracy
y_pred = model.predict(X_test)

# The model outputs probabilities, so let's threshold at 0.5:
y_pred = (y_test > 0.5)
accuracy = np.mean(y_test == y_pred)

print("Accuracy on test data is %.2f %%" % (100 * accuracy))

