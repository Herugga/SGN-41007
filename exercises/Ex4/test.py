# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:26:18 2019

@author: Mikko
"""

# Training code:
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Load sample data
X = np.loadtxt("X.csv", delimiter = ",")
y = np.loadtxt("y.csv")

# Convert from indices into categorical form.
# Note: More general approach is:
# y = tf.keras.utils.to_categorical(y)
y = (y == 1)

# Split to training and evaluation:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(y_train.shape)
# First we initialize the model. "Sequential" means there are no loops.
clf = tf.keras.models.Sequential()

# Add layers one at the time. Each with 100 nodes.
clf.add(tf.keras.layers.Dense(100, input_dim=2, activation = 'sigmoid'))
clf.add(tf.keras.layers.Dense(100, activation = 'sigmoid'))
clf.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

# The code is compiled to CUDA or C++
clf.compile(loss='mean_squared_error', optimizer='sgd')
clf.fit(X_train, y_train, epochs=20, batch_size=16) # takes a few seconds

# Check accuracy
y_pred = clf.predict(X_test)

# The model outputs probabilities, so let's threshold at 0.5:
y_pred = (y_test > 0.5)
accuracy = np.mean(y_test == y_pred)

print("Accuracy on test data is %.2f %%" % (100 * accuracy))