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

import MNIST_v2_functions as func
from sklearn.datasets import fetch_openml

            #### MAIN EXECUTABLE ####  

if __name__ == '__main__':

    # Load in MNIST 784 Dataset 
    print("Loading MNIST...")
    MNIST = fetch_openml(name='mnist_784',version=1,return_X_y=False)
    X_train,y_train = MNIST['data'][:10000],MNIST['target'][:10000]
    X_test,y_test = MNIST['data'][10000:20000],MNIST['target'][10000:20000]
   
    """ Control Classifiers """
    print("Training Control Classifiers...")
    for I in range (0,1,1):
        layers = (20,20,20,20)
        control_clf = func.Train_MLP_Model('Control '+str(I),layers=layers,
                                            X=X_train,y=y_train,seed=None)
        control_predicitions = func.Test_MLP_Model(control_clf,X_test)
        control_confmat = func.Confusion_Matrix(control_clf,
                                y_test,control_predicitions,show=True)