"""
Landon Buell
Qioayan Yu
What-to-Approximate Scheme
18 August 2020
"""

        #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import time
import tensorflow.keras as keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import WTA_Utilities as utils

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    homePath = os.getcwd()
    outputPath = os.path.join(os.path.dirname(homePath),'Raw_Data')
    outputBuffer = os.path.join(outputPath,utils.FrameName)

    outputFrame = pd.DataFrame(data=None,columns=utils.FrameCols)
    outputFrame.to_csv(path_or_buf=outputBuffer,mode='w')

    N_iters = 2
   
    X_train,y_train,X_test,y_test = utils.CIFAR10()
    y_train = keras.utils.to_categorical(y_train,10)
    y_test = keras.utils.to_categorical(y_test,10)

    for SIZE in utils.KERNELSIZES:
        print("\tTesting Kernel Size:",SIZE)
        likeModelData = np.array([])        # hold data for similar models

        for i in range (N_iters):
            NETWORK = utils.NeuralNetwork('SINGLE_'+str(SIZE),SIZE)
            NETWORK.__TRAIN__(X_train,y_train)
            scores = NETWORK.__EVALUATE__(X_test,y_test)

            likeModelData = np.append(likeModelData,scores)
            

        likeModelData = likeModelData.reshape(N_iters,-1)      
        averageScores = np.mean(likeModelData,axis=0)
        exportData = np.array([NETWORK.modelName])
        exportData = np.append(exportData,averageScores)
    
        frame = pd.DataFrame(data=[exportData],columns=utils.FrameCols)
        frame.to_csv(path_or_buf=outputBuffer,mode='a',header=False)


    print("=)")

