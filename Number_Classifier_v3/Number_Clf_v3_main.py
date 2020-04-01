"""
Landon Buell
Qiaoyan Yu
MNIST Tests v3 - Main
31 March 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import sys
import time

import Number_Clf_v3_functions as func
from sklearn.datasets import fetch_openml

            #### MAIN EXECUTABLE ####  

if __name__ == '__main__':

    # Load in MNIST 784 Dataset 
    print("Loading MNIST...")
    MNIST = fetch_openml(name='mnist_784',version=1,return_X_y=False)
    print(sys.getsizeof(MNIST))
    X_train,y_train = MNIST['data'][:10000],MNIST['target'][:10000]
    X_test,y_test = MNIST['data'][10000:15000],MNIST['target'][10000:15000]
   
    """ Run Classifiers """
    N_iters = 100
    test_type = 'round_to_0_v1'            # current test type    
    layers = (100,100)                    # Layers in MLP
    output_path = str(layers)           # output path name
    output_matrix = np.array([])        # matrix to export
    print("Training Classifiers...")    # messege to user

    for I in range (0,N_iters,1):      
        
        t_0 = time.process_time()           # start time
        clf_model = func.Train_MLP_Model(str(test_type)+'_'+str(I),
                        layers=layers,X=X_train,y=y_train,seed=None)
        t_f = time.process_time()           # end time

        y_pred = clf_model.predict(X_test)              # run predicitions
        confmat = func.Confusion_Matrix(clf_model,
                            y_test,y_pred,show=False)   # build confusion matrix
        scores = func.metric_scores(y_test,y_pred)      # compute precision & recall scores
        dt = np.round(t_f-t_0,8)                        # compute time to train model

        print('\tIteration',str(I),'time:',dt)          # message to user

        row = np.array([clf_model.name,dt,layers,
                        clf_model.loss_,clf_model.n_iter_]) # row of data to export
        row = np.append(row,scores)         # add 20 scores to output
        row = np.append(row,confmat)        # add 100 elements of confusion matrix to output
        output_matrix = np.append(output_matrix,row)
    
    
    output_matrix = output_matrix.reshape(N_iters,125)   # matrix to output
    frame = func.create_dataframe(output_matrix)        # convert to pd dataframe
    try:
        frame.to_csv(output_path+ '/' +str(test_type)+'.csv')   # write frame
    except:
        os.mkdir(output_path)                                   # create output path
        frame.to_csv(output_path+ '/' +str(test_type)+'.csv')   # write frame
    print("Program Time:",time.process_time())                  # total process time
        