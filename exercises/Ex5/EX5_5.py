# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:02:44 2019

@author: Mikko
"""
import tensorflow 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import cv2

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()



if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = tensorflow.image.grayscale_to_rgb(
    tensorflow.convert_to_tensor(x_train)
)
x_train = tensorflow.image.resize(
x_train,
[128,128]
)
x_test = tensorflow.image.grayscale_to_rgb(
tensorflow.convert_to_tensor(x_test)
)
x_test = tensorflow.image.resize(
x_test,
[128,128]
)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

base_model = tensorflow.keras.applications.mobilenet.MobileNet(input_shape = (128,128,3), alpha= 0.25, include_top = False)
base_model.summary()

in_tensor = base_model.inputs[0]# Grab the input of base model
out_tensor = base_model.outputs[0]# Grab the output of base model


out_tensor = tensorflow.keras.layers.Flatten()(out_tensor)
out_tensor = tensorflow.keras.layers.Dense(100, activation = "sigmoid")(out_tensor)
out_tensor = tensorflow.keras.layers.Dense(2, activation = "sigmoid")(out_tensor)

# Define the full model by the endpoints.
model = tensorflow.keras.Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])