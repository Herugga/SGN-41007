# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:29:14 2019

@author: Mikko
"""

import numpy as np
import matplotlib.pyplot as plt


 # Q3 #
def log_loss(w, X, y):
    
    L = 0 # Accumulate loss terms here.
    # Process each sample in X:
    for n in range(X.shape[0]):
        L += np.log(1 + np.exp(-y[n] * np.dot(w, X[n])))
    return L
    
def grad(w, X, y):

    G = 0

    for n in range(X.shape[0]):
        numerator = np.exp(-y[n] * np.dot(w, X[n])) * (-y[n]) * X[n]
        denominator = 1 + np.exp(-y[n] * np.dot(w, X[n]))

        G += numerator / denominator

    return G
    
if __name__ == "__main__":
    # 1) Load X and y.    
    X=np.loadtxt('X.csv',delimiter=',')
    y=np.loadtxt('y.csv',delimiter=',')
    # 2) Initialize w at w = np.array([1, -1])
    w = np.array([1, -1])
    # 3) Set step_size to a small positive value.
    step_size=0.01
    # 4) Initialize empty lists for storing the path and
    W = []
    accuracies = []
    for iteration in range(100):

        # 5) Apply the gradient descent rule.
        w=w-step_size*grad(w,X,y)
        # 6) Print the current state.
        print ("Iteration %d: w = %s (log-loss = %.2f)" % \
              (iteration, str(w), log_loss(w, X, y)))
        # 7) Compute the accuracy (already done for you)
        # Predict class 1 probability
        y_prob = 1 / (1 + np.exp(-np.dot(X, w)))
                # Threshold at 0.5 (results are 0 and 1)
        y_pred = (y_prob > 0.5).astype(int)
                # Transform [0,1] coding to [-1,1] coding
        y_pred = 2*y_pred - 1
        accuracy = np.mean(y_pred == y)
        accuracies.append(accuracy)
        W.append(w)
        
    W = np.array(W)
    plt.figure(figsize = [5,5])
    plt.subplot(211)
    plt.plot(W[:,0], W[:,1], 'ro-')
    plt.xlabel('w$_0$')
    plt.ylabel('w$_1$')
    plt.title('Optimization path')
    
    plt.subplot(212)
    plt.plot(100.0 * np.array(accuracies), linewidth = 2)
    plt.ylabel('Accuracy / %')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig("log_loss_minimization.pdf", bbox_inches = "tight")

