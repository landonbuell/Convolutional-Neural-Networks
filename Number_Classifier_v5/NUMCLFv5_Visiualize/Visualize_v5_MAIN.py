"""
Landon Buell
Qiaoyan Yu
Number Classifier v4 - main
3 April 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import time

import Visualize_v5_Utilities as Vis_utils

if __name__ == '__main__':

    int_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v5/NUMCLFv5_Visualize'
    out_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Presentations/OTSACS'
    data_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v5/Raw_Data'
    CSV_FILES = ['Baseline_v1.csv','Mute_MSB_v4.csv','Mute_MSB_v3.csv','Mute_MSB_v2.csv','Mute_MSB_v1.csv']

    files_objs = []
    for file in CSV_FILES:
        inst = Vis_utils.filedata(str(file),data_path,file)
        inst = inst.split_X()
        files_objs.append(inst)     # add instance to list


    os.chdir(out_path)
    labs=['Baseline','0.5% Trigger','5% Trigger','25% Trigger','50% Trigger']

    # Loss Function values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Avg_Loss',labs=labs,
                        ylab='Loss Function Value',title="Single Hidden Layer Loss Value",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Avg_Loss',labs=labs,
                        ylab='Loss Function Value',title="Double Hidden Layer Loss Value",save=True)

    # Precision Score Values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Avg_Prec',labs=labs,
                        ylab='Precision Score',title="Single Hidden Layer Precision",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Avg_Prec',labs=labs,
                        ylab='Precision Score',title="Double Hidden Layer Precision ",save=True)
    
    # Recall Score Values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Avg_Recall',labs=labs,
                        ylab='Recall Score',title="Single Hidden Layer Recall",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Avg_Recall',labs=labs,
                        ylab='Recall Score',title="Double Hidden Layer Recall",save=True)