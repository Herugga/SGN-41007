# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:43:46 2019
@author: Mikko
"""

# -*- coding: utf-8 -*-

import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from simplelbp import local_binary_pattern
from sklearn.model_selection import cross_val_score

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

def extract_lbp_features(X, P = 8, R = 5):
    """
    Extract LBP features from all input samples.
    - R is radius parameter
    - P is the number of angles for LBP
    """
    
    F = [] # Features are stored here
    
    N = X.shape[0]
    for k in range(N):
        
        print("Processing image {}/{}".format(k+1, N))
        
        image = X[k, ...]
        lbp = local_binary_pattern(image, P, R)
        hist = np.histogram(lbp, bins=range(257))[0]
        F.append(hist)

    return np.array(F)

# Test our loader

X, y = load_data(".")
F = extract_lbp_features(X)
print("X shape: " + str(X.shape))
print("F shape: " + str(F.shape))

# Continue your code here...

## LOOP FOR Classifiers
Model = []
KNN = sklearn.neighbors.KNeighborsClassifier()
LDA = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
SVC1 = sklearn.svm.SVC(kernel ='linear')
LR = sklearn.linear_model.LogisticRegression(solver='liblinear',multi_class='ovr')

Model.append(KNN)
Model.append(LDA)
Model.append(SVC1)
Model.append(LR)


for i in range(len(Model)):
    model = Model[i]
    print(np.mean(cross_val_score(model,F, y, cv=5)))
    

