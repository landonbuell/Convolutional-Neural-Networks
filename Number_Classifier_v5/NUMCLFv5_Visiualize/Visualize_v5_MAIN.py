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
    CSV_FILES = ['Baseline_v1.csv','Mute_MSB_v1.csv','Mute_MSB_v2.csv']

    files_objs = []
    for file in CSV_FILES:
        inst = Vis_utils.filedata(str(file),data_path,file)
        inst = inst.split_X()
        files_objs.append(inst)     # add instance to list


    os.chdir(out_path)
    ylabs=['Baseline','50/50 Trigger','25/75 Trigger',#'05/95 Trigger'
           ]

    # Loss Function values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Avg_Loss',ylabs=ylabs,
                        title="Single Hidden Layer Loss Value",show=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Avg_Loss',ylabs=ylabs,
                        title="Double Hidden Layer Loss Value",show=True)

    # Precision Score Values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Avg_Prec',ylabs=ylabs,
                        title="Single Hidden Layer Precision Score",show=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Avg_Prec',ylabs=ylabs,
                        title="Double Hidden Layer Precision Score",show=True)
    
    # Recall Score Values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Avg_Recall',ylabs=ylabs,
                        title="Single Hidden Layer Recall Score",show=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Avg_Prec',ylabs=ylabs,
                        title="Double Hidden Layer Recall Score",show=True)