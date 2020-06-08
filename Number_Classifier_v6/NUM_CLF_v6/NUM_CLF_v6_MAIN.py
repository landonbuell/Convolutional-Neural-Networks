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

    MODEL = utils.Keras_Model('layers')

    # Training
    history = MODEL.fit(X_train,y_train,batch_size=128,epochs=20,verbose=10)
    utils.Plot_History(history,MODEL,show=True)

    # Evaluate