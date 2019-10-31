# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:13:52 2019

@author: Oma
"""

import numpy as np
import matplotlib.pyplot as plt

n = np.arange(100)
f = 0.015
x = np.sin(2*np.pi*f*n) + np.sqrt(0.3)*np.random.randn(100)

plt.plot(n,x)

scores = []
frequencies = []

for f in np.linspace(0, 0.5, 1000):
    # Create vector e. Assume data is in x.
    n = np.arange(100)
    z = -2*np.pi*1j*f*n
    e = np.exp(z)
   
    score = np.abs(np.dot(x,e)) # <compute abs of dot product of x and e>
    scores.append(score)
    frequencies.append(f)
    
fHat = frequencies[np.argmax(scores)]
