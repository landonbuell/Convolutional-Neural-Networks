"""
Landon Buell
Qiaoyan Yu
Number Classifier v4 - Analysis
3 April 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import time

import Number_Clf_v4_analysis_utilities as analysis_utils

            #### MAIN ANALYSIS PROGRAM ####

if __name__ == '__main__':

        
    """ Collect All Data """
    print("Collecting Data...\n")

    PLOTTED_VALUES = ['Train Time','Loss Value','Iterations']
    MODEL_LAYER_SIZES = analysis_utils.N_layer_models    # model sizes dictionary
    TEST_TYPES = ['control_v1','round_0_decimals_v1','round_0_decimals_v2']
    NODE_SIZES = [20,40,60,80,100,120]
    

    home_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v4'

    # Index through training time, loss func value, iterations to converge
    for value in PLOTTED_VALUES:
        
        # index through single,double,quadruple layer models
        for key in MODEL_LAYER_SIZES:               # each layer size:      
            work_path = home_path + '/' + key       # working path for data  
            os.chdir(work_path)                     # change to path
  
            # index through control, round v1, round v2
            for test_type in TEST_TYPES:            # for the type of test conducted
                # indec


                for neurons in NODE_SIZES:          # nodes per layer
                    filename = test_type + '_' + str(neurons)
                    filedata = pd.read_csv(filename+'csv',usecols=value)     
                    

                    avgs = np.append(avgs,np.average(filedata))     # add average to arr
                    mini = np.min(filedata)                         # add minimum to arr
                    maxi = np.max(filedata)                         # add maximum to arr