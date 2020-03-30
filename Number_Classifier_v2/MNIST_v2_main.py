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
    N_iters = 100
    test_type = 'Control_v1'               # current test type
    output_matrix = np.array([])        # matrix to export
    layers = (100,100)                  # Layers in MLP
    print("Training Classifiers...")    # messege to user

    for I in range (0,N_iters,1):      
        
        t_0 = time.process_time()
        clf_model = func.Train_MLP_Model(str(test_type)+'_'+str(I),
                        layers=layers,X=X_train,y=y_train,seed=None)
        t_f = time.process_time()

        y_pred = clf_model.predict(X_test)
        confmat = func.Confusion_Matrix(clf_model,
                            y_test,y_pred,show=True)
        scores = func.metric_scores(y_test,y_pred)
        dt = np.round(t_f-t_0,8)

        print('\tIteration',str(I),'time:',dt)

        row = np.array([clf_model.name,dt,clf_model.loss_,clf_model.n_iter_])
        row = np.append(row,scores) 
        output_matrix = np.append(output_matrix,row)
             
    output_matrix = output_matrix.reshape(N_iters,24)
    frame = func.create_dataframe(output_matrix)
    frame.to_csv(str(test_type)+'.csv')