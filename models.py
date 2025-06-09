# Creates the baseline models for replicating the results in
#     Aimone, James B., William Severa, and J. Darby Smith. 
#     "Synaptic Sampling of Neural Networks." 2023 IEEE International 
#     Conference on Rebooting Computing (ICRC). IEEE, 2023.
# In practice, you should probably load/create your keras model as desired as these 
# are just toy models.  
# 
# Note kernel_contstraint is set to clip the weight values.

import keras

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import layers, metrics

from constraint import MinMax


def load_model(model, data, input_dim_size, input_shape, output_classes):
    if(model == 'ff'):
        initializer=keras.initializers.RandomUniform(minval=.25, maxval=.75)
        model = Sequential(
            [
                layers.Dense(512, input_dim=input_dim_size, activation='relu', kernel_initializer=initializer, kernel_constraint=MinMax([0.0, 1.0])),
                layers.Dropout(0.2), 
                layers.Dense(512, activation='relu', kernel_initializer=initializer, kernel_constraint=MinMax([0.0, 1.0])),
                layers.Dropout(0.2), 
                layers.Dense(output_classes, activation='softmax', kernel_initializer=initializer, kernel_constraint=MinMax([0.0, 1.0])),
            ]
        )
        model.summary()
    elif(model == 'ff_w'):
        initializer=keras.initializers.RandomUniform(minval=.25, maxval=.75)
        model = Sequential(
            [
                layers.Dense(1024, input_dim=input_dim_size, activation='relu', kernel_initializer=initializer, kernel_constraint=MinMax([0.0, 1.0])),
                layers.Dropout(0.2), 
                layers.Dense(512, activation='relu', kernel_initializer=initializer, kernel_constraint=MinMax([0.0, 1.0])),
                layers.Dropout(0.2),
                layers.Dense(output_classes, activation='softmax', kernel_initializer=initializer, kernel_constraint=MinMax([0.0, 1.0])),
            ]
        )
        model.summary()
    elif(model == 'ff_nodropout'):
        initializer=keras.initializers.RandomUniform(minval=.25, maxval=.75)
        model = Sequential(
            [
                layers.Dense(400, input_dim=input_dim_size, activation='relu', kernel_initializer=initializer, kernel_constraint=MinMax([0.0, 1.0])),
                layers.Dense(output_classes, activation='softmax', kernel_initializer=initializer, kernel_constraint=MinMax([0.0, 1.0])),
            ]
        )
        model.summary()
    elif(model == 'conv' and (data == 'mnist' or data == 'fashion')):
        initializer=keras.initializers.RandomUniform(minval=.25, maxval=.75)
        model = Sequential(
            [
                layers.Conv2D(32, (3, 3), input_shape = input_shape, activation='relu', kernel_initializer = initializer, kernel_constraint = MinMax([0.0, 1.0])),
                layers.MaxPooling2D((2,2)),
                layers.Flatten(),
                layers.Dropout(0.5), 
                layers.Dense(100,activation='relu', kernel_initializer=initializer, kernel_constraint=MinMax([0.0, 1.0])),
                layers.Dropout(0.5), 
                layers.Dense(output_classes, activation='softmax', kernel_initializer=initializer, kernel_constraint=MinMax([0.0, 1.0])),
            ]
        )
        model.summary()
    elif(model == 'conv' and (data=='cifar10' or data == 'cifar100')):
        initializer=keras.initializers.RandomUniform(minval=-0.99, maxval=.99)
        model = Sequential(
            [
                layers.Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu'),#, kernel_initializer = initializer, kernel_constraint=MinMax([0.0, 1.0])),
                layers.MaxPooling2D((2,2)),
                layers.Conv2D(64, (3,3), activation = 'relu'),#, kernel_initializer = initializer, kernel_constraint=MinMax([0.0, 1.0])),
                layers.MaxPooling2D((2,2)),
                layers.Conv2D(128, (3,3), activation = 'relu'),#, kernel_initializer = initializer, kernel_constraint=MinMax([0.0, 1.0])),
                layers.MaxPooling2D((2,2)),
                layers.Flatten(),
                layers.Dropout(0.25), 
                layers.Dense(256, activation = 'relu'),#, kernel_initializer = initializer, kernel_constraint=MinMax([-1.0, 1.0])),#, kernel_initializer = initializer, kernel_constraint=MinMax([0, 1.0])),
                layers.Dropout(0.25), 
                layers.Dense(128, activation = 'relu', kernel_initializer = initializer, kernel_constraint=MinMax([-1.0, 1.0])),
                layers.Dropout(0.25), 
                layers.Dense(output_classes, activation='softmax', kernel_initializer=initializer, kernel_constraint=MinMax([-1.0, 1.0])),

            ]
        )
        model.summary()
    elif(model == 'conv_raw' and (data=='cifar10' or data == 'cifar100')):
        model = Sequential(
            [
                layers.Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu'),#, kernel_initializer = initializer, kernel_constraint=MinMax([0.0, 1.0])),
                layers.MaxPooling2D((2,2)),
                layers.Conv2D(64, (3,3), activation = 'relu'),#, kernel_initializer = initializer, kernel_constraint=MinMax([0.0, 1.0])),
                layers.MaxPooling2D((2,2)),
                layers.Conv2D(128, (3,3), activation = 'relu'),#, kernel_initializer = initializer, kernel_constraint=MinMax([0.0, 1.0])),
                layers.MaxPooling2D((2,2)),
                layers.Flatten(),
                layers.Dropout(0.5), 
                layers.Dense(256, activation = 'relu'),#, kernel_initializer = initializer, kernel_constraint=MinMax([0.0, 1.0])),
                layers.Dropout(0.5), 
                layers.Dense(128, activation = 'relu'),#, kernel_initializer = initializer, kernel_constraint=MinMax([0.0, 1.0])),
                layers.Dropout(0.5), 
                layers.Dense(output_classes, activation='softmax'),#, kernel_initializer=initializer, kernel_constraint=MinMax([0.0, 1.0])),

            ]
        )
        model.summary()
    return model