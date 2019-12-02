# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:46:52 2019

@author: Mikko
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:02:44 2019

@author: Mikko

 ## DONE! ##
"""

import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFECV
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

#X_train, y_train, X_test, y_test = loa

mat = loadmat("arcene.mat")
X_train = mat["X_train"]
y_train = mat["y_train"].ravel()
X_test = mat["X_test"]
y_test = mat["y_test"].ravel()


solver =  sklearn.linear_model.LogisticRegression(solver="liblinear")
slc = RFECV(estimator=solver, step=50, verbose=1)
slc.fit(X_train,y_train)
print("Selectet Features",np.sum(slc.support_))

print("Optimal number of features : %d" % slc.n_features_)


y_hat = slc.predict(X_test)
accuracy = accuracy_score(y_test, y_hat)

print("Accuracy is ",accuracy*100,"%")


# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(slc.grid_scores_) + 1), slc.grid_scores_)
plt.show()
