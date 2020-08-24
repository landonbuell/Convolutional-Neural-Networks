"""
Landon Buell
Qioayan Yu
What-To-Approximate Visualizations
23 August 2020
"""

        #### IMPORTS ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

        #### Variables ####

filenames = ["Baseline.csv","Approx100.csv","Approx200.csv","Approx300.csv"]
parentPath = "C:\\Users\\Landon\\Documents\\GitHub\\Convolutional-Neural-Networks\\What-To-Approximate\\Raw_Data"

legendLabels = ["Baseline","Approximate 100 Pixels","Approximate 200 Pixels","Approximate 300 Pixels"]


        #### CLASS DECLARATIONS ####

class PlotMetrics :
    """
    Class to produce visualizations of metric Datat
    """
    def __init__(self,dataobjects,labels):
        """ Initialize Class Object Instance """
        self.dataobjs = dataobjects
        self._labels = labels
        self._kernelSizes = np.arange(2,7)
   
    def PlotLossScore (self,name,save=False,show=True):
        """ Visualize Loss Scores """
        plt.figure(figsize=(20,16))
        plt.xlabel("Pixel Grouping Size",size=60,weight='bold')
        plt.ylabel("Loss Function Value",size=60,weight='bold')

        plt.plot(self._kernelSizes,self.dataobjs[0]._losses,color='red',marker='o',markersize=24,label=self._labels[0])
        plt.plot(self._kernelSizes,self.dataobjs[1]._losses,color='blue',marker='^',markersize=24,label=self._labels[1])
        plt.plot(self._kernelSizes,self.dataobjs[2]._losses,color='purple',marker='^',markersize=24,label=self._labels[2])
        plt.plot(self._kernelSizes,self.dataobjs[3]._losses,color='green',marker='^',markersize=24,label=self._labels[3])

        plt.xticks(self._kernelSizes,['2 x 2','3 x 3','4 x 4','5 x 5','6 x 6'],size=40)
        plt.yticks(np.arange(1.75,4.75,0.25),size=40)
        plt.grid()
        plt.legend(loc=1,fontsize=30)

        if save == True:
            plt.savefig(str(name)+'.png')
        if show == True:
            plt.show()

    def PlotPrecisionScore (self,name,save=False,show=True):
        """ Visualize Loss Scores """
        plt.figure(figsize=(20,16))
        plt.xlabel("Pixel Grouping Size",size=60,weight='bold')
        plt.ylabel("Precision Score",size=60,weight='bold')

        plt.plot(self._kernelSizes,self.dataobjs[0]._precisions,color='red',marker='o',markersize=24,label=self._labels[0])
        plt.plot(self._kernelSizes,self.dataobjs[1]._precisions,color='blue',marker='^',markersize=24,label=self._labels[1])
        plt.plot(self._kernelSizes,self.dataobjs[2]._precisions,color='purple',marker='^',markersize=24,label=self._labels[2])
        plt.plot(self._kernelSizes,self.dataobjs[3]._precisions,color='green',marker='^',markersize=24,label=self._labels[3])

        plt.xticks(self._kernelSizes,['2 x 2','3 x 3','4 x 4','5 x 5','6 x 6'],size=40)
        plt.yticks(np.arange(0.2,0.9,0.1),size=40)
        plt.grid()
        plt.legend(loc=1,fontsize=30)

        if save == True:
            plt.savefig(str(name)+'.png')
        if show == True:
            plt.show()

    def PlotRecallScore (self,name,save=False,show=True):
        """ Visualize Loss Scores """
        plt.figure(figsize=(20,16))
        plt.xlabel("Pixel Grouping Size",size=60,weight='bold')
        plt.ylabel("Recall Score",size=60,weight='bold')

        plt.plot(self._kernelSizes,self.dataobjs[0]._recalls,color='red',marker='o',markersize=24,label=self._labels[0])
        plt.plot(self._kernelSizes,self.dataobjs[1]._recalls,color='blue',marker='^',markersize=24,label=self._labels[1])
        plt.plot(self._kernelSizes,self.dataobjs[2]._recalls,color='purple',marker='^',markersize=24,label=self._labels[2])
        plt.plot(self._kernelSizes,self.dataobjs[3]._recalls,color='green',marker='^',markersize=24,label=self._labels[3])

        plt.xticks(self._kernelSizes,['2 x 2','3 x 3','4 x 4','5 x 5','6 x 6'],size=40)
        plt.yticks(np.arange(0.2,0.9,0.1),size=40)
        plt.grid()
        plt.legend(loc=1,fontsize=30)

        if save == True:
            plt.savefig(str(name)+'.png')
        if show == True:
            plt.show()

class FileData :
    """ Class to Organize Data From CS File """

    def __init__(self,dataframe,file):
        """ Initialize Class Object Instance """
        self._modelnames = dataframe['Model Name'].to_numpy()
        self._losses = dataframe['Loss'].to_numpy()
        self._precisions = dataframe['Precision'].to_numpy()
        self._recalls = dataframe['Recall'].to_numpy()
        self.filename = file
