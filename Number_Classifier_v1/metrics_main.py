"""
Landon Buell
Prof. Yu
Metric Responses - Main
10 January 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import metrics_functions as func

from sklearn.datasets import fetch_openml

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    """ Loading in MNIST Data set """
    print("\tCollecting Data...")
    MNIST = fetch_openml('mnist_784',version=1)     # collect dataset
    X,y = MNIST['data'],MNIST['target']             # isolate data & labels
    X,y = X[:10000],y[:10000]
    labels = np.arange(0,10,1)                      # labels in MNIST
    stddev = 2**6
    num = 1
    output_matrix = np.array([])

    for I in np.arange(0,num,1):
    
        """ Split Training & Testing Data Sets """
        print("\tSplitting Data...")
        trainpts,testpts = func.split_train_test(len(y),0.2)
        xtrain,ytrain = X[trainpts],y[trainpts]         # create training data
        xtest,ytest = X[testpts],y[testpts]             # create testing data
        ntest,ntrain = len(xtest),len(ytest)            # num pts in each array

        """ Control Classifier """
        print("\tRunning Control SGD Classifier:") 
        CLF,ypreds = func.SGD_CLF('control',xtrain,ytrain,xtest)    # train & test clf
        scores = func.classification_scores(ytest,ypreds,labels)    # evaluate
        confmat = func.confusion_matrix(CLF,ytest,ypreds,labels,show=True)
        scores = np.append(scores,confmat)
        output_matrix = np.append(output_matrix,scores)

        """ Noisy-Clean Classifier """
        print("\tRunning Noisy-Clean SGD Classifier:") 
        xtrain_noise = func.add_noise(xtrain,0,stddev)        # add noise to train data
        CLF,ypreds = func.SGD_CLF('noisy-clean',xtrain_noise,ytrain,xtest)
        scores = func.classification_scores(ytest,ypreds,labels)
        confmat = func.confusion_matrix(CLF,ytest,ypreds,labels,show=True)
        scores = np.append(scores,confmat)
        output_matrix = np.append(output_matrix,scores)

        """ Clean-Noisy Classifier """
        print("\tRunning Clean-Noisy SGD Classifier:") 
        xtest_noise = func.add_noise(xtest,0,stddev)          # add noise to testing data
        CLF,ypreds = func.SGD_CLF('clean-noisy',xtrain,ytrain,xtest_noise)
        scores = func.classification_scores(ytest,ypreds,labels)
        confmat = func.confusion_matrix(CLF,ytest,ypreds,labels,show=True)
        scores = np.append(scores,confmat)
        output_matrix = np.append(output_matrix,scores)
    
        """ Noisy Classifier """
        print("\tRunning Noisy SGD Classifier:") 
        CLF,ypreds = func.SGD_CLF('clean-noisy',xtrain_noise,ytrain,xtest_noise)
        scores = func.classification_scores(ytest,ypreds,labels)
        confmat = func.confusion_matrix(CLF,ytest,ypreds,labels,show=True)
        scores = np.append(scores,confmat)
        output_matrix = np.append(output_matrix,scores)

    ncols = 103
    output_matrix = output_matrix.reshape(-1,ncols)
    print(time.process_time())
