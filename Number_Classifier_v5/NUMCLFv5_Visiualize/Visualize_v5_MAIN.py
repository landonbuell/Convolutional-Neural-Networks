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
    CSV_FILES = ['Baseline_v1.csv','Mute_MSB_v1.csv']

    BASELINE = Vis_utils.filedata('Baseline',data_path,CSV_FILES[0])
    BASELINE = BASELINE.split_X()
    MUTE_MSB = Vis_utils.filedata('Mute MSB',data_path,CSV_FILES[1])
    MUTE_MSB = MUTE_MSB.split_X()

    os.chdir(out_path)

    Vis_utils.Plot_Metrics([BASELINE,MUTE_MSB],'single_layer',title="Single Hidden Layer Models",save=True)
    Vis_utils.Plot_Metrics([BASELINE,MUTE_MSB],'double_layer',title="Double Hidden Layer Models",save=True)
    Vis_utils.Plot_Metrics([BASELINE,MUTE_MSB],'triple_layer',title="Triple Hidden Layer Models",save=True)
    Vis_utils.Plot_Metrics([BASELINE,MUTE_MSB],'quadruple_layer',title="Quadruple Hidden Layer Models",save=True)

    