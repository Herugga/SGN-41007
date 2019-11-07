

from sklearn import neighbors
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Q5 #

# A # 
mat = loadmat("twoClassData.mat")
X = mat["X"]
Y = mat["y"].ravel()
XY = list(zip(X, Y))
random.Random(100).shuffle(XY)
X,Y = zip(*XY)


# B # 
Xtrain=X[0:200]
Ytrain=Y[0:200]
Xtest=X[200:]
Ytest=Y[200:]

# C # 
model=neighbors.KNeighborsClassifier(n_neighbors=8, leaf_size=1)
model.fit(Xtrain,Ytrain)
Ypredict1=model.predict(Xtest)
print(accuracy_score(Ytest,Ypredict1))

# D # 
clf = LinearDiscriminantAnalysis()
clf.fit(Xtrain,Ytrain)
Ypredict2=clf.predict(Xtest)
print(accuracy_score(Ytest,Ypredict2))