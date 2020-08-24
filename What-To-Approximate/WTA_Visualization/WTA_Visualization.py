"""
Landon Buell
Qioayan Yu
What-To-Approximate Visualizations
23 August 2020
"""

        #### IMPORTS ####

import numpy as np
import pandas as pd
import os

import WTA_Visualization_Utils as utils

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':


    dataObjs = []
    for file in utils.filenames:
        frame = pd.read_csv(os.path.join(utils.parentPath,file))
        dataObj = utils.FileData(frame,file)
        dataObjs.append(dataObj)

    Metrics = utils.PlotMetrics(dataObjs,utils.legendLabels)
    Metrics.PlotLossScore("WTA_LossScores",save=True)
    Metrics.PlotPrecisionScore("WTA_PrecisionScores",save=True)
    Metrics.PlotRecallScore("WTA_RecallScores",save=True)

    print("=)")
