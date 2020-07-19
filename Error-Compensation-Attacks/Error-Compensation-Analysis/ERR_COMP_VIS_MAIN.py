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
    data_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Error-Compensation-Attacks/Raw_Data_v1'
    expt_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Presentations/Error-Comp-Attacks-v1'

    CSV_FILES = [   'Baseline.csv','Approx4.csv','Approx6.csv','Approx8.csv'
                 
                 ]

    files_objs = []
    for file in CSV_FILES:
        inst = Vis_utils.filedata(str(file),data_path,file)
        inst = inst.split_X()
        files_objs.append(inst)     # add instance to list

    os.chdir(expt_path)

    labs=[  'Baseline Model',
          '4 Pixel Approximation',
          '6 Pixel Approximation',
          '8 Pixel Approximation'
          ]

    # Loss Function values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Average Loss',labs=labs,
                        ylab='Loss Function Value',title="Loss Value - One Layer Group",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Average Loss',labs=labs,
                        ylab='Loss Function Value',title="Loss Value - Two Layer Groups",save=True)

    # Precision Score Values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Average Precision',labs=labs,
                        ylab='Precision Score',title="Precision Score - One Layer Group",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Average Precision',labs=labs,
                        ylab='Precision Score',title="Precision Score - Two Layer Groups",save=True)

    # Recall Score Values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Average Recall',labs=labs,
                        ylab='Recall Score',title="Recall Score - One Layer Group",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Average Recall',labs=labs,
                        ylab='Recall Score',title="Recall Score - Two Layer Groups",save=True)
    
    # Executation Time Values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Average Train Time',labs=labs,
                        ylab='Time to Train',title="Train Time - One Layer Group",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Average Train Time',labs=labs,
                        ylab='Time to Train',title="Train Time - Two Layer Groups",save=True)