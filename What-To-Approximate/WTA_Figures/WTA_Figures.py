"""
Landon Buell
Qioayan Yu
What-to-Approximate Scheme
25 August 2020
"""

        #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import time
import tensorflow.keras as keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import WTA_Figures_Utils as utils

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    classNames = ['airplane','automobile','bird','cat','deer',
                    'dog','frog','horse','ship','truck']

    X_train,y_train,X_test,y_test = utils.CIFAR10() 
    y_train = keras.utils.to_categorical(y_train,10)

    WTA_Layer = utils.WhatToApproximateLayer(maskSize=200)

    Network = utils.NeuralNetwork("JARVIS",3)
    Network.__TRAIN__(X_train[:10000],y_train[:10000])

    for i in range (0,10,1):

        OriginalImage = X_test[i]
        ApproximateImage = WTA_Layer.Call(OriginalImage)

        print("Class Label:",classNames[y_test[i][0]])
        prediction = Network.model.predict(np.array([OriginalImage]).reshape(1,32,32,3))
        prediction = np.argmax(prediction,axis=-1)[0]
        print("\tOriginal Prediction:",classNames[prediction])
        prediction = Network.model.predict(np.array([ApproximateImage]).reshape(1,32,32,3))
        prediction = np.argmax(prediction,axis=-1)[0]
        print("\tApproximate Prediction:",classNames[prediction])

        utils.PlotSample(OriginalImage,"OriginalImage"+str(i),True,False)
        utils.PlotSample(ApproximateImage,"ApproximateImage"+str(i),True,False)

        


        



