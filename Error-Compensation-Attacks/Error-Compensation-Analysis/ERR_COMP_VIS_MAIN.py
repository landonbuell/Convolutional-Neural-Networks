"""
Landon Buell
Qioyan Yu
Error-Comp-Visulization-v0
24 June 2020
"""
      
            #### IMPORTS ####

import numpy as np
import pandas as pd
import os

import ERR_COMP_VIS_Utilities as Vis_utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # Initialize Params
    home_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Error-Compensation-Attacks/Error-Compensation-Analysis'
    data_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Error-Compensation-Attacks/Raw_Data'
    expt_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Presentations/Error-Compensation-Attacks/'

    CSV_FILES = [   'Baseline.csv','Approx_2.csv','Approx_4.csv',
                    'Approx_6.csv','Approx_8.csv','Approx_10.csv']

    files_objs = []
    for file in CSV_FILES:
        inst = Vis_utils.filedata(str(file),data_path,file)
        inst = inst.split_X()
        files_objs.append(inst)     # add instance to list

    os.chdir(expt_path)

    """
    for I in range(0,10,2):
        X = X[:10]
        Vis_utils.Plot_Matrix(X[I],'Original: '+str(y[I]),save=True)     
        # 3 pixel border
        Approx_Layer = Vis_utils.ApproximationLayer(rows=np.concatenate((np.arange(0,3),np.arange(25,28))),
                                                    cols=np.concatenate((np.arange(0,3),np.arange(25,28))))
        X = Approx_Layer.call(X)
        Vis_utils.Plot_Matrix(X[I],'3 Pixel Border: '+str(y[I]),save=True)
        # 5 pixel border
        Approx_Layer = Vis_utils.ApproximationLayer(rows=np.concatenate((np.arange(0,5),np.arange(23,28))),
                                                    cols=np.concatenate((np.arange(0,5),np.arange(23,28))))
        X = Approx_Layer.call(X)
        Vis_utils.Plot_Matrix(X[I],'5 Pixel Border: '+str(y[I]),save=True)
        # 7 pixel border
        Approx_Layer = Vis_utils.ApproximationLayer(rows=np.concatenate((np.arange(0,7),np.arange(21,28))),
                                                    cols=np.concatenate((np.arange(0,7),np.arange(21,28))))
        X = Approx_Layer.call(X)
        Vis_utils.Plot_Matrix(X[I],'7 Pixel Border: '+str(y[I]),save=True)

    """

    labs=[  'Baseline','2 Pixel Border','4 Pixel Border',
            '6 Pixel Border','8 Pixel Border','10 Pixel Border']

    # Loss Function values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Average Loss',labs=labs,
                        ylab='Loss Function Value',title="Single Hidden Layer Loss Value",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Average Loss',labs=labs,
                        ylab='Loss Function Value',title="Double Hidden Layer Loss Value",save=True)

    # Precision Score Values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Average Precision',labs=labs,
                        ylab='Precision Score',title="Single Hidden Layer Precision",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Average Precision',labs=labs,
                        ylab='Precision Score',title="Double Hidden Layer Precision",save=True)
    
    # Recall Score Values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Average Recall',labs=labs,
                        ylab='Recall Score',title="Single Hidden Layer Recall",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Average Recall',labs=labs,
                        ylab='Recall Score',title="Double Hidden Layer Recall",save=True)

 