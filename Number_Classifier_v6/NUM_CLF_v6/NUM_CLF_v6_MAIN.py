"""
Landon Buell
Number Classifier v6
Main Script
7 June 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import time
import os

import tensorflow.keras as keras

import NUM_CLF_v6_Utilities as utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    home_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v6/NUM_CLF_v6'
    data_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v6/Raw_Data'

    X_train,X_test,y_train,y_test = utils.Load_MNIST()
    y_train = keras.utils.to_categorical(y_train,10)

    print("X train shape:",X_train.shape)
    print("X test shape:",X_test.shape)
    print("y train shape:",y_train.shape)
    print("y test shape:",y_test.shape)

    MODEL = utils.Keras_Model(layers=[40,40],
                              rows=np.arange(0,14),cols=np.arange(0,14))

    X = X_train[0:4]

    approx = utils.ApproximationLayer(rows=np.arange(0,14),cols=np.arange(0,14))

    Y = approx.call(before)

    for I in range (0,4):
        utils.Plot_Matrix(X[I])
        utils.Plot_Matrix(Y[I])

    # Training
    #history = MODEL.fit(X_train,y_train,batch_size=128,epochs=20,verbose=10)
    #utils.Plot_History(history,MODEL,show=True)
