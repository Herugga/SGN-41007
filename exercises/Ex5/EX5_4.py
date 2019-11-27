# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:02:44 2019

@author: Mikko
"""
import tensorflow

base_model = tensorflow.keras.applications.mobilenet.MobileNet(input_shape = (128,128,3), alpha= 0.25, include_top = False)
#base_model.summary()

in_tensor = base_model.inputs[0]# Grab the input of base model
out_tensor = base_model.outputs[0]# Grab the output of base model


out_tensor = tensorflow.keras.layers.Flatten()(out_tensor)
out_tensor = tensorflow.keras.layers.Dense(100, activation = "sigmoid")(out_tensor)
out_tensor = tensorflow.keras.layers.Dense(2, activation = "sigmoid")(out_tensor)

# Define the full model by the endpoints.
model = tensorflow.keras.models.Model(inputs  = [in_tensor],outputs = [out_tensor])

# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd')
model.summary()
