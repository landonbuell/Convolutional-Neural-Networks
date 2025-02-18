"""
Landon Buell
Qioayan Yu
CLF v6 Visualize - MAIN
15 June 2020
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import NUM_VIS_v6_Utilities as Vis_utils

        #### MAIN EXECUTBALE ####

if __name__ == '__main__':

    home_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v6'
    data_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v6/Raw_Data'
    expt_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v6/Figures'

    CSV_FILES = ['Baseline.csv','Approx_3.csv','Approx_5.csv','Approx_7.csv',]

    files_objs = []
    for file in CSV_FILES:
        inst = Vis_utils.filedata(str(file),data_path,file)
        inst = inst.split_X()
        files_objs.append(inst)     # add instance to list

    os.chdir(expt_path)
    

    # Plot Images Before & After
    X,y = Vis_utils.Load_MNIST()
    

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
    labs=['Baseline','3 Pixel Border','5 Pixel Border','7 Pixel Border',]

    # Loss Function values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Avg_Loss',labs=labs,
                        ylab='Loss Function Value',title="Single Hidden Layer Loss Value",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Avg_Loss',labs=labs,
                        ylab='Loss Function Value',title="Double Hidden Layer Loss Value",save=True)

    # Precision Score Values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Avg_Prec',labs=labs,
                        ylab='Precision Score',title="Single Hidden Layer Precision",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Avg_Prec',labs=labs,
                        ylab='Precision Score',title="Double Hidden Layer Precision",save=True)
    
    # Recall Score Values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Avg_Recall',labs=labs,
                        ylab='Recall Score',title="Single Hidden Layer Recall",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Avg_Recall',labs=labs,
                        ylab='Recall Score',title="Double Hidden Layer Recall",save=True)

    """    