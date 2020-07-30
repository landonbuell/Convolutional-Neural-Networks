"""
Landon Buell
Qioyan Yu
Error-Comp-Timing-Utilities
29 July 2020
"""
        
            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

import tensorflow as tf
import tensorflow.keras as keras

import ERR_COMP_TIMING_Utils as utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    X_train,y_train,X_test,y_test = utils.Load_CIFAR10()
    

