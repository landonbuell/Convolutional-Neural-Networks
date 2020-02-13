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

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    """ Loading in MNIST Data set """
    print("\tCollecting Data...")
    X,y,labels = func.collect_MNIST()   # collect MNISt data set
    stddev = 2**6                       # std dev
    niters = 2                          # number of iterations
    output_matrix = np.array([])

    """ Split Training & Testing Data Sets """
    print("\tSplitting Data...")
    trainpts,testpts = func.split_train_test(len(y),0.2)
    xtrain,ytrain = X[trainpts],y[trainpts]         # create training data
    xtest,ytest = X[testpts],y[testpts]             # create testing data
    ntest,ntrain = len(xtest),len(ytest)            # num pts in each array

    for I in range (0,niters,1):
        """ Control Classifier """
        name = 'Control'+'_v'+str(I)
        print("\tRunning",name,"SGD Classifier:")       
        CLF,ypreds = func.SGD_CLF(name,xtrain,ytrain,xtest)    # train & test clf
        scores = func.classification_scores(ytest,ypreds,labels)    # evaluate
        confmat = func.confusion_matrix(CLF,ytest,ypreds,labels,show=False)

    for I in range (0,niters,1):
        """ Noisy-Clean Classifier """
        name = 'Noisy-Clean'+'_v'+str(I)
        print("\tRunning",name,"SGD Classifier:")     
        xtrain_noise = func.add_noise(xtrain,0,stddev)        # add noise to train data
        CLF,ypreds = func.SGD_CLF(name,xtrain_noise,ytrain,xtest)
        scores = func.classification_scores(ytest,ypreds,labels)
        confmat = func.confusion_matrix(CLF,ytest,ypreds,labels,show=False)
   
    for I in range (0,niters,1):
        """ Clean-Noisy Classifier """
        name = 'Clean-Noisy'+'_v'+str(I)
        print("\tRunning",name,"SGD Classifier:") 
        xtest_noise = func.add_noise(xtest,0,stddev)          # add noise to testing data
        CLF,ypreds = func.SGD_CLF(name,xtrain,ytrain,xtest_noise)
        scores = func.classification_scores(ytest,ypreds,labels)
        confmat = func.confusion_matrix(CLF,ytest,ypreds,labels,show=False)
    
    for I in range (0,niters,1):
        """ Noisy Classifier """
        name = 'Noisy'+'_v'+str(I)
        print("\tRunning",name,"SGD Classifier:") 
        CLF,ypreds = func.SGD_CLF(name,xtrain_noise,ytrain,xtest_noise)
        scores = func.classification_scores(ytest,ypreds,labels)
        confmat = func.confusion_matrix(CLF,ytest,ypreds,labels,show=False)

    ncols = 104
    output_matrix = output_matrix.reshape(-1,ncols)
    dataframe = func.assemble_dataframe(output_matrix,len(labels))
    dataframe.to_csv('Metrics_Test_v1.txt',sep='\t',
                     header=True,index=True)
    print(dataframe)
    print(time.process_time())
