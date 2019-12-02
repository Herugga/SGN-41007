# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:37:24 2019

@author: Mikko

"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
import numpy as np

# Select features
clf = LogisticRegression(penalty = "l1",solver="liblinear")
C_range = 10.0 ** np.arange(-5,6, 0.5)

accuracies = []
nonzeros = []
bestScore = 0

mat = loadmat("arcene.mat")
X_train = mat["X_train"]
y_train = mat["y_train"].ravel()
X_test = mat["X_test"]
y_test = mat["y_test"].ravel()
     
   
from sklearn.model_selection import GridSearchCV

param_test1 ={'C':[1,10,50,100,1000,10000,100000]}  
clf=LogisticRegression(solver='liblinear',penalty='l1')
gsearch1= GridSearchCV(estimator =clf,param_grid =param_test1,scoring='roc_auc',cv=5)  
gsearch1.fit(X_train,y_train)  
print(gsearch1.best_params_, gsearch1.best_score_ )
clf.fit(X_train,y_train)
print(len(np.nonzero(clf.coef_)[0]))
y_predict=clf.predict(X_test)
print('CLF -',accuracy_score(y_test,y_predict)) 

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:02:44 2019

@author: Mikko

 ## TODO: PRINTING?? ##
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
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.hist(importances[indices], bins='auto')
plt.show()

plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()


features = [range(0,100,1)]
indices = np.argsort(importances)[-400:]  # top n features
plt.figure()
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

plt.bar(np.arange(len(importances)), importances)
plt.show()