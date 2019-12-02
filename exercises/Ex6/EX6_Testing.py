# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:02:44 2019

@author: Mikko

 ## TODO: UNTESTED! ##
"""
import tensorflow 
import cv2
from sklearn.ensemble import  ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
import random

#def load_arcene():
#    mat = loadmat("arcene.mat")
#    X = mat["X"]
#    Y = mat["y"].ravel()
#    XY = list(zip(X, Y))
#    random.Random(100).shuffle(XY)
#    X,Y = zip(*XY)
    
    

# Load Arcene data; 100+100 samples with dimension 10000:
# Mass spectrometer measurements from ovarian cancer patients and healthy controls.
#X_train, y_train, X_test, y_test = loa

mat = loadmat("arcene.mat")
X_train = mat["X_train"]
y_train = mat["y_train"].ravel()
X_test = mat["X_test"]
y_test = mat["y_test"].ravel()

classifiers = [(RandomForestClassifier(), "Random Forest"),
               (ExtraTreesClassifier(), "Extra-Trees"),
               (AdaBoostClassifier(), "AdaBoost"),
               (GradientBoostingClassifier(), "GB-Trees")]

for clf, name in classifiers:
    clf.n_estimators = 100
    accuracies = [] 
    for iteration in range(100):
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_hat)
        accuracies.append(accuracy)

#classifiers = [(RandomForestClassifier(), "Random Forest")]#,
##               (ExtraTreesClassifier(), "Extra-Trees"),
##               (AdaBoostClassifier(), "AdaBoost"),
##               (GradientBoostingClassifier(), "GB-Trees")]
#
#for clf, name in classifiers:
#    clf.n_estimators = 100
#    
#    accuracies = [] 
#    for iteration in range(100):
#        clf.fit(X_train, y_train)
#        y_hat = clf.predict(X_test)
#        accuracy = accuracy_score(y_test, y_hat)
#        accuracies.append(accuracy)

