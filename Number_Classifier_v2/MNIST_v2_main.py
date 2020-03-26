"""
Landon Buell
Qiaoyan Yu
Approximate Comp v1
24 March 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import time

import MNIST_v2_functions as func
from sklearn.datasets import fetch_openml

            #### MAIN EXECUTABLE ####  

if __name__ == '__main__':

    # Load in MNIST 784 Dataset 
    print("Loading MNIST...")
    MNIST = fetch_openml(name='mnist_784',version=1,return_X_y=False)
    X_train,y_train = MNIST['data'][:10000],MNIST['target'][:10000]
    X_test,y_test = MNIST['data'][10000:20000],MNIST['target'][10000:20000]
   
    """ Run Classifiers """
    N_iters = 10
    test_type = 'Control'               # current test type
    datamatrix = np.array([])           # matrix to hold all data
    layers = (20,20,20,20)
    print("Training Classifiers...")

    for I in range (0,N_iters,1):      
        
        t_0 = time.process_time()
        clf_model = func.Train_MLP_Model(str(test_type)+'_'+str(I),
                        layers=layers,X=X_train,y=y_train,seed=None)
        t_f = time.process_time()

        y_pred = clf_model.predict(X_test)
        confmat = func.Confusion_Matrix(clf_model,
                                y_test,y_pred,show=False)
        scores = func.metric_scores(y_test,y_pred)
        dt = np.round(t_f-t_0,8)

        row = np.array([clf_model.name,dt])
        row = np.append(row,scores)  
        datamatrix = np.append(datamatrix,row)
        
        print('\tIteration',str(I),'time:',dt)
        
    datamatrix = datamatrix.reshape(N_iters,22)
    frame = func.create_dataframe(datamatrix)
    frame.to_csv(str(test_type)+'.csv')