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
    TEST_TYPES = ['control_v1','round_0_decimals_v1','round_0_decimals_v2','gaussian_noise_v1']
    NODE_SIZES = [20,40,60,80,100,120]

    FILEDATA_INSTANCES = []
    
    cols = analysis_utils.cols_to_use
    frame_cols = analysis_utils.out_frame_idx
    outname = 'model_analysis'
    home_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v4'
    out_path = home_path + '/Number_Classifier_v4_Analysis'
    
        
    # index through single,double,quadruple layer models
    for key in MODEL_LAYER_SIZES:               # each layer size:      
        work_path = home_path + '/' + key       # working path for data  
        os.chdir(work_path)                     # change to path
  
        # index through control, round v1, round v2
        for test_type in TEST_TYPES:            # for the type of test conducted
 
            for neurons in NODE_SIZES:          # nodes per layer
                filename = test_type + '_' + str(neurons)
                dataframe = pd.read_csv(filename+'.csv',usecols=cols)
                filedata = analysis_utils.file_data(filename,dataframe,
                                                    key,test_type,neurons)
                del(dataframe)
                FILEDATA_INSTANCES.append(filedata) # add to list

    #### We now have the results of all tets stored in a list of class instances
    print(len(FILEDATA_INSTANCES))
    print(time.process_time())

    data = np.array([])                     # data to convert to dataframe
    for file in FILEDATA_INSTANCES:         # for each file that we just read
        data = np.append(data,file.data_for_file())
    data = data.reshape(len(FILEDATA_INSTANCES),-1)
    print(data.shape)
    idx_list = [x.index_for_file() for x in FILEDATA_INSTANCES]

    output_frame = pd.DataFrame(data=data,index=idx_list,columns=frame_cols)
    output_frame.to_csv(out_path+'/'+outname+'.csv')