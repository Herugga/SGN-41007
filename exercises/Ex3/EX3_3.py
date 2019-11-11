# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:03:45 2019

@author: Mikko
"""

import matplotlib.pyplot as plt
import sklearn.model_selection

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score


digits = load_digits()
plt.gray()
plt.imshow(digits.images[0])
plt.show()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(digits.data, digits.target, test_size=0.2)

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
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    print('model',accuracy_score(y_test,predict))
    




