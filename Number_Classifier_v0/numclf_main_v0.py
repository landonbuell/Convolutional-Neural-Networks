"""
Landon Buell
Number Classifier v0
Main Executable Functions
27 December 2019
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import numclf_func_v0 as numclf

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    print("Start")
                     

    print("Splitting Data Set:")
    xydict = numclf.split_train_test(xdata,ydata)

    print("Training Classifier:")
    number_clf = SGDClassifier(random_state=0)
    number_clf.fit(xydict['X_train'],xydict['Y_train'])

    print("Building Confusion Matrix:")
    confmat = numclf.confusion(number_clf,xydict['X_train'],xydict['Y_train'])
