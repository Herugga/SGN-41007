
import matplotlib.pyplot as plt
import numpy as np

## EX2_Q4 ##


# A #
x=np.arange(0,901,1)
n=np.arange(500,601,1)
y1=np.zeros(500)
y2=np.cos(2 * np.pi * 0.1 * n)
y3=np.zeros(300)
y=np.concatenate((y1,y2,y3))

# B #
y_n = y + np.sqrt(0.5) * np.random.randn(y.size)


 # C & D # 
f0 = 0.1
h = np.exp(-2 * np.pi * 1j * f0 * n) #f0 set as 0.3 what should be?

y_dd = (np.convolve(h, y_n, 'same'))
y_sd = np.abs(np.convolve(h, y_n, 'same'))


fig, ax = plt.subplots(4, 1) # Create a figure with 4 axes
ax[0].plot(y) # This will be the topmost axis
ax[1].plot(y_n) # This will be the second axis
ax[2].plot(y_dd) # This will be the 3rd axis
ax[3].plot(y_sd) # This will be the 4th axis
plt.show() # Display on screen.