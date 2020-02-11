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
    print("\tCollecting Data...")
    MNIST = fetch_openml('mnist_784',version=1)     # collect dataset
    X,y = MNIST['data'],MNIST['target']             # isolate data & labels
    labels = np.arange(0,10,1)                      # labels in MNIST

    """ Split Training & Testing Data Sets """
    print("\tSplitting Data...")
    trainpts,testpts = func.split_train_test(len(y),0.2)
    xtrain,ytrain = X[trainpts],y[trainpts]         # create training data
    xtest,ytest = X[testpts],y[testpts]             # create testing data

    """ Control Classifier """
    print("\tRunning Control SGD Classifier:")
    control_classifier = func.SGD_CLF('control',xtest,ytest,0)
    control_predicitons = func.CLF_Predict(control_classifier,xtest)
    control_scores = func.classification_scores(ytest,control_predicitons,
                                                labs=labels,avg=None)
    control_confmat = func.confusion_matrix(control_classifier,ytest,
                                            control_predicitons,labels,show=True)