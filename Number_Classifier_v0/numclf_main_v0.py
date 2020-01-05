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
import numclf_func_v0 as numclf

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    mnist = fetch_openml('mnist_784',version=1)     # MNIST DATA Set
    print(mnist.keys())
    xdata,ydata = mnist['data'],mnist['target']      # data & labels
    ydata = ydata.astype(np.uint8)                 

    number_clf,xydict = numclf.SGD_Classifier(xdata,ydata,1000)
