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

    MODEL_LAYER_SIZES = analysis_utils.N_layer_models    # model sizes dictionary
    TEST_TYPES = ['control_v1','round_0_decimals_v1','round_0_decimals_v2']
    NODE_SIZES = [20,40,60,80,100,120]
    DATA_DICTIONARY = {}                        # will hold all class instances

    home_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v4'

    for key in MODEL_LAYER_SIZES:               # each layer size:      
        work_path = home_path + '/' + key       # working path for data  
        os.chdir(work_path)                     # change to path
        for test_type in TEST_TYPES:            # each type of test            
            for node_size in NODE_SIZES:        # each size layer:
                filename = test_type + '_' + str(node_size)         
                # Read to data into a frame & make instance of it
                dataframe = pd.read_csv(filename+'.csv')
                dataset = analysis_utils.dataset(filename,dataframe,node_size)
                del(dataframe)
                DATA_DICTIONARY.update({str(key)+'_'+filename:dataset})  

    for key in DATA_DICTIONARY.keys():
        print(key)

    """ Analyze Single layer Models """
    NAMES = ['single_layer_models_control_v1_20','single_layer_models_control_v1_40','single_layer_models_control_v1_60'
                'single_layer_models_control_v1_80','single_layer_models_control_v1_100','single_layer_models_control_v1_120']
    analysis_utisl.Analyze_model(NAMES,DATA_DICTIONARY)


