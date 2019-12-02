# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:02:44 2019

@author: Mikko

 ## DONE ##
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.io import loadmat
from sklearn.metrics import accuracy_score


mat = loadmat("arcene.mat")
X_train = mat["X_train"]
y_train = mat["y_train"].ravel()
X_test = mat["X_test"]
y_test = mat["y_test"].ravel()



clf = RandomForestClassifier(n_estimators=100)   
accuracies = []
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_hat)
print("Accuracy",accuracy)

importances = clf.feature_importances_


plt.bar(np.arange(len(importances)), importances)
plt.show()