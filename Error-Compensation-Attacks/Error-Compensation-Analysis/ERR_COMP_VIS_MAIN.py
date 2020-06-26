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

    CSV_FILES = [   'Baseline.csv','Approx_4.csv','Comp_4.csv',
                                    'Approx_8.csv','Comp_8.csv']

    files_objs = []
    for file in CSV_FILES:
        inst = Vis_utils.filedata(str(file),data_path,file)
        inst = inst.split_X()
        files_objs.append(inst)     # add instance to list

    os.chdir(expt_path)

    """
    X_train,y_train,X_test,y_test = Vis_utils.Load_Fashion_MNIST10(10,100)
    X = X_test

    Vis_utils.Plot_Matrix(X[1],'Original',save=True)
    ApproxLayer = Vis_utils.ApproximationLayer(rows=Vis_utils.approx_index4,
                                                cols=Vis_utils.approx_index4)
    X = ApproxLayer.call(X)                 
    Vis_utils.Plot_Matrix(X[1],'Approx4',save=True)
    CompLayer = Vis_utils.CompensationLayer(rows=Vis_utils.approx_index4,
                                            cols=Vis_utils.approx_index4)
    X = CompLayer.call(X)
    Vis_utils.Plot_Matrix(X[1],'Compensate4',save=True)

    """
    labs=[  'Baseline','4 Pixel Attack','4 Pixel Compensation',
                        '8 Pixel Attack','8 Pixel Compensation']

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

