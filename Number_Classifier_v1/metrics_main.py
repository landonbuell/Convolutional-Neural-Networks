"""
Landon Buell
Prof. Yu
Metric Responses - Main
10 January 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import time
import metrics_functions as func

from sklearn.datasets import fetch_openml

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    """ Loading in MNIST Data set """
    print("Preparing Data...")
    MNIST = fetch_openml('mnist_784',version=1)     # collect dataset
    X,y = MNIST['data'],MNIST['target']             # isolate data & labels
    trainpts,testpts = func.split_train_test(len(y),0.1)
    xtrain,ytrain = X[trainpts],y[trainpts]         # create training data
    xtest,ytest = X[testpts],y[testpts]             # create testing data

    """ Control Classifier """
    print("Creating Control SGD Classifier:")
    control_classifier = func.SGD_CLF('control',xtest,ytest,0)