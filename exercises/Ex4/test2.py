# Training code

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

# We use the handwritten digit database "MNIST".
# 60000 training and 10000 test images of
# size 28x28
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Keras assumes 4D input, but MNIST is lacking color channel.
# -> Add a dummy dimension at the end.

X_train = X_train[..., np.newaxis] / 255.0
X_test  = X_test[..., np.newaxis] / 255.0

# Output has to be one-hot-encoded
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

num_featmaps = 32   # This many filters per layer
num_classes = 10    # Digits 0,1,...,9
num_epochs = 50     # Show all samples 50 times
w, h = 5, 5         # Conv window size

model = Sequential()

# Layer 1: needs input_shape as well.
model.add(Conv2D(num_featmaps, (w, h),
          input_shape=(28, 28, 1),
          activation = 'relu'))

# Layer 2:
model.add(Conv2D(num_featmaps, (w, h), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3: dense layer with 128 nodes
# Flatten() vectorizes the data:
# 32x10x10 -> 3200
# (10x10 instead of 14x14 due to border effect)
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))

# Layer 4: Last layer producing 10 outputs.
model.add(Dense(num_classes, activation='softmax'))

# Compile and train
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 10,
          validation_data = (X_test, y_test))