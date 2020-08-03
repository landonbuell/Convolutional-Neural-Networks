"""
Landon Buell
Qioyan Yu
Error-Comp-Figures
20 July 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import ERR_COMP_FIGS_Utilities as utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    
        #### PLOT METRICS ####

    infile = '4Pixels.xlsx'
    filedata = pd.read_excel(infile,index_col=0)
    print(filedata.head())

    # Plot Loss Metrics
    cols = ['Baseline Loss','Approximated Loss','33% Blurred Loss','50% Blurred Loss','Compensated Loss']
    data = [filedata[x] for x in cols]
    utils.Plot_Metrics('4 Pixel Border Loss','Loss Function Value',data,
                       'loss',True,False)

    # Plot Precision Metrics
    cols = ['Baseline Precision','Approximated Precision',
            '33% Blurred Precision','50% Blurred Precision','Compensated Precision']
    data = [filedata[x] for x in cols]
    utils.Plot_Metrics('4 Pixel Border Precision Score','Precision Score Value',data,
                       'precision',True,False)

    # Plot Precision Metrics
    cols = ['Baseline Recall','Approximated Recall',
            '33% Blurred Recall ', '50% Blurred Recall ','Compensated Recall']
    data = [filedata[x] for x in cols]
    utils.Plot_Metrics('4 Pixel Border Recall Score','Recall Score Value',data,
                       'recall',True,False)
    """
            #### PLOT EXECUTION TIME PERCENTAGE ####

    infile = 'ErrComp_TimeDiff.xlsx'
    filedata = pd.read_excel(infile,index_col=0)
    print(filedata.head())

    dt = []
    baseline = np.array(filedata['Baseline'],dtype=np.float64)
    filedata = filedata.drop(['Model','Baseline'],axis=1)
    for col in filedata.columns:

        # compute averages
        diff = utils.percent_diff(baseline,np.array(filedata[col],dtype=np.float64))
        dt.append(diff)        # add the percent diff

    utils.Plot_PercentDiff('Fitting Time Difference from Baseline','Percentage Difference',
                           labs=filedata.columns,ydata=dt)
    
    """
            #### PLOT COMEPNSATION vs. EXECUTION TIME ####
    path = 'C:\\Users\\Landon\\Documents\\GitHub\\Convolutional-Neural-Networks\\' + \
                'Error-Compensation-Attacks\\Raw_Data_Timing'
    infiles = ['Compensate2time.csv','Compensate4time.csv',
               'Compensate6time.csv','Compensate8time.csv']
    plot_data = np.array([])

    for file in infiles:
        filedata = pd.read_csv(os.path.join(path,file),index_col=0)
        filedata = filedata.drop(['Model'],axis=1)
        executionTime = filedata.to_numpy(dtype=np.float64)[:,0]
        compensationTime = filedata.to_numpy(dtype=np.float64)[:,1]
        layerDiff = 100*(compensationTime/executionTime)
        plot_data = np.append(plot_data,layerDiff)

    plot_data = plot_data.reshape(len(infiles),-1)
    labels = ['2 Pixel Border','4 Pixel Border','6 Pixel Border','8 Pixel Border']
    utils.Plot_PercentDiff2('Compensation in Forward Pass','Percentage of Time',
                            labels,plot_data,show=True)


        
