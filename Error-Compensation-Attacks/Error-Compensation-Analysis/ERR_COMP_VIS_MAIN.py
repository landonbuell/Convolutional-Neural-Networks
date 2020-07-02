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

    CSV_FILES = [   'Baseline.csv',
                 'Approx_2.csv','Approx_4.csv','Approx_6.csv','Approx_8.csv',
                 'Comp_2.csv','Comp_4.csv','Comp_6.csv','Comp_8.csv'
                 ]

    files_objs = []
    for file in CSV_FILES:
        inst = Vis_utils.filedata(str(file),data_path,file)
        inst = inst.split_X()
        files_objs.append(inst)     # add instance to list

    os.chdir(expt_path)

    """
    X_train,y_train,X_test,y_test = Vis_utils.Load_CIFAR10()
    X = X_test[:10]

    Vis_utils.Plot_Matrix(X[1],'_OriginalShip',save=True)
    Vis_utils.Plot_Matrix(X[9],'_OriginalAuto',save=True)

    ApproxLayer2 = Vis_utils.ApproximationLayer(rows=Vis_utils.approx_index2,
                                                cols=Vis_utils.approx_index2)
    X2 = ApproxLayer2.call(X)
    Vis_utils.Plot_Matrix(X2[1],'_2ApproxShip',save=True)
    Vis_utils.Plot_Matrix(X2[9],'_2ApproxAuto',save=True)

    ApproxLayer4 = Vis_utils.ApproximationLayer(rows=Vis_utils.approx_index4,
                                                cols=Vis_utils.approx_index4)
    X4 = ApproxLayer4.call(X)
    Vis_utils.Plot_Matrix(X4[1],'_4ApproxShip',save=True)
    Vis_utils.Plot_Matrix(X4[9],'_4ApproxAuto',save=True)

    CompLayer2 = Vis_utils.CompensationLayer(rows=Vis_utils.approx_index2,
                                                cols=Vis_utils.approx_index2)
    X2 = CompLayer2.call(X2)
    Vis_utils.Plot_Matrix(X2[1],'_2CompShip',save=True)
    Vis_utils.Plot_Matrix(X2[9],'_2CompAuto',save=True)

    CompLayer4 = Vis_utils.CompensationLayer(rows=Vis_utils.approx_index4,
                                                cols=Vis_utils.approx_index4)
    X4 = CompLayer2.call(X4)
    Vis_utils.Plot_Matrix(X2[1],'_4CompShip',save=True)
    Vis_utils.Plot_Matrix(X2[9],'_4CompAuto',save=True)
    """

    labs=[  'Baseline Model',
          '2 Pixel Approximation','4 Pixel Approximation',
          '6 Pixel Approximation','8 Pixel Approximation',
          '2 Pixel Compensation', '4 Pixel Compensation',
          '6 Pixel Compensation', '8 Pixel Compensation'
          ]

    # Loss Function values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Average Loss',labs=labs,
                        ylab='Loss Function Value',title="Loss Value - One Layer Group",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Average Loss',labs=labs,
                        ylab='Loss Function Value',title="Loss Value - Two Layer Groups",save=True)
    Vis_utils.Plot_Metric(files_objs,'triple_layer',metric='Average Loss',labs=labs,
                        ylab='Loss Function Value',title="Loss Value - Three Layer Groups",save=True)
    # Precision Score Values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Average Precision',labs=labs,
                        ylab='Precision Score',title="Precision Score - One Layer Group",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Average Precision',labs=labs,
                        ylab='Precision Score',title="Precision Score - Two Layer Groups",save=True)
    Vis_utils.Plot_Metric(files_objs,'triple_layer',metric='Average Precision',labs=labs,
                        ylab='Precision Score',title="Precision Score - Three Layer Groups",save=True)

    # Recall Score Values
    Vis_utils.Plot_Metric(files_objs,'single_layer',metric='Average Recall',labs=labs,
                        ylab='Recall Score',title="Recall Score - One Layer Group",save=True)
    Vis_utils.Plot_Metric(files_objs,'double_layer',metric='Average Recall',labs=labs,
                        ylab='Recall Score',title="Recall Score - Two Layer Groups",save=True)
    Vis_utils.Plot_Metric(files_objs,'triple_layer',metric='Average Recall',labs=labs,
                        ylab='Recall Score',title="Recall Score - Three Layer Groups",save=True)
