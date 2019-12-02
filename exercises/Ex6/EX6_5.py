# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:37:24 2019

@author: Mikko

"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
#from sklearn.metrics import accuracy_score
from scipy.io import loadmat
import numpy as np

# Select features
clf = LogisticRegression(penalty = "l1",solver="liblinear")
C_range = 10.0 ** np.arange(-5,6, 0.5) # or np.arange(-5,6, 0.2)

accuracies = []
nonzeros = []
bestScore = 0

mat = loadmat("arcene.mat")
X_train = mat["X_train"]
y_train = mat["y_train"].ravel()
X_test = mat["X_test"]
y_test = mat["y_test"].ravel()
     
   
for C in C_range:

    clf.C = C
    #print(C)
    clf.fit(X_train, y_train)
    y_prediction = clf.predict(X_test)
    x_accuracy = 100.0 * np.mean(y_prediction == y_test)
    cv_score = 100.0 * np.mean(cross_val_score(clf,X_train,y_train,cv=5))

    print("Accuracy of y_pred: ",x_accuracy) #"Accuracy of y_test ",y_accuracy)
    print("Cross_Val_Score",cv_score)
    print("Num of Cofficences",len(np.nonzero(clf.coef_)[0]))
