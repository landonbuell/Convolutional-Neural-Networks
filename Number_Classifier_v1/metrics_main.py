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
    print(np.shape(X))
    labels = np.arange(0,10,1)                      # labels in MNIST
    dfcols = np.array(['Avg Precision','Avg Recall','Avg F1'])
    dfcols = np.append(dfcols,np.arange(0,len(labels)**2)).astype(str)
    score_Dataframe = pd.DataFrame(columns=dfcols)  
    
    """ Split Training & Testing Data Sets """
    print("\tSplitting Data...")
    trainpts,testpts = func.split_train_test(len(y),0.2)
    xtrain,ytrain = X[trainpts],y[trainpts]         # create training data
    xtest,ytest = X[testpts],y[testpts]             # create testing data

    """ Control Classifier """
    print("\tRunning Control SGD Classifier:") 
    CLF,ypreds = func.SGD_CLF('control',xtrain,ytrain,xtest)    # train & test clf
    scores = func.classification_scores(ytest,ypreds,labels)    # evaluate
    print("\t\tControl Scores:",scores)
    confmat = func.confusion_matrix(CLF,ytest,ypreds,labels,show=True)

    """ Noisy-Clean Classifier """
    print("\tRunning Noisy-Clean SGD Classifier:") 
    xtrain_noise = func.add_noise(xtrain,0,2**5)            # add noise to train data
    CLF,ypreds = func.SGD_CLF('noisy-clean',xtrain_noise,ytrain,xtest)
    scores = func.classification_scores(ytest,ypreds,labels)
    print("\t\tNosiy-Clean Scores:",scores)
    confmat = func.confusion_matrix(CLF,ytest,ypreds,labels,show=True)

    """ Clean-Noisy Classifier """
    print("\tRunning Clean-Noisy SGD Classifier:") 
    xtest_noise = func.add_noise(xtest,0,2**5)            # add noise to testing data
    CLF,ypreds = func.SGD_CLF('clean-noisy',xtrain,ytrain,xtest_noise)
    scores = func.classification_scores(ytest,ypreds,labels)
    print("\t\tClean-Noisy Scores:",scores)
    confmat = func.confusion_matrix(CLF,ytest,ypreds,labels,show=True)
    
    """ Noisy Classifier """
    print("\tRunning Noisy SGD Classifier:") 
    CLF,ypreds = func.SGD_CLF('clean-noisy',xtrain_noise,ytrain,xtest_noise)
    scores = func.classification_scores(ytest,ypreds,labels)
    print("\t\tNoisy Scores:",scores)
    confmat = func.confusion_matrix(CLF,ytest,ypreds,labels,show=True)

    print(time.process_time())
