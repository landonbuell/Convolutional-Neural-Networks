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

            #### DEFINITIONS ####

def Plot_Metrics (title,ylab,ydata,metric,save=False,show=True):
    """
    Plot Metrics From a Given Dataframe
    --------------------------------
    title (str) : Title for Figure
    ylab (str) : Label for y-axis
    ydata (iter) : List of 4 arrays to plot
    metric (str) : Name of Metric Being Plotted
    --------------------------------
    Return None
    """
    plt.figure(figsize=(20,12))
    plt.title(title,size=60,weight='bold',pad=20)
    plt.ylabel(ylab,size=40,weight='bold')
    plt.xlabel('Kernel Shape',size=40,weight='bold')

    kernel_sides = np.array([2,3,4,5,6])

    plt.plot(kernel_sides,ydata[0],color='red',linestyle='-',marker='o',ms=20,label='Baseline')
    plt.plot(kernel_sides,ydata[1],color='blue',linestyle='-',marker='^',ms=20,label='Approximated')
    plt.plot(kernel_sides,ydata[2],color='magenta',linestyle='-',marker='s',ms=20,label='33% Blurred')
    plt.plot(kernel_sides,ydata[3],color='green',linestyle='-',marker='s',ms=20,label='50% Blurred')
    plt.plot(kernel_sides,ydata[4],color='gray',linestyle='-',marker='v',ms=20,label='Compensated')
    
    if metric in ['precision','recall']:
        plt.yticks(np.arange(0,1.1,0.1),size=35)
    else:
        plt.yticks(size=35)
    plt.xticks(kernel_sides,['2x2','3x3','4x4','5x5','6x6'],size=40)
    plt.grid()
    plt.legend(fontsize=25,loc=0)    
      
    if save == True:
        title = title.replace(' ','_')
        plt.savefig(title+'.png')
    if show == True:
        plt.show()
    plt.close()

def Plot_PercentDiff (title,ylab,labs,ydata,save=False,show=True):
    """
    Plot Metrics From a Given Dataframe
    --------------------------------
    title (str) : Title for Figure
    ylab (str) : Label for y-axis
    ydata (iter) : List of 4 arrays to plot
    --------------------------------
    Return None
    """
    plt.figure(figsize=(20,12))
    plt.title(title,size=60,weight='bold',pad=20)
    plt.ylabel(ylab,size=40,weight='bold')
    plt.xlabel('Kernel Shape',size=40,weight='bold')

    kernel_sides = np.array([2,3,4,5,6])
    plt.hlines(0,2,7,color='black')

    plt.plot(kernel_sides,ydata[0],color='blue',linestyle='-',marker='^',ms=20,label=labs[0])
    plt.plot(kernel_sides,ydata[1],color='green',linestyle='-',marker='^',ms=20,label=labs[1])
    plt.plot(kernel_sides,ydata[2],color='gray',linestyle='-',marker='^',ms=20,label=labs[2])
    plt.plot(kernel_sides,ydata[3],color='purple',linestyle='-',marker='^',ms=20,label=labs[3])

    plt.plot(kernel_sides,ydata[4],color='orange',linestyle='--',marker='s',ms=20,label=labs[4])
    plt.plot(kernel_sides,ydata[5],color='yellow',linestyle='--',marker='s',ms=20,label=labs[5])
    plt.plot(kernel_sides,ydata[6],color='red',linestyle='--',marker='s',ms=20,label=labs[6])
    plt.plot(kernel_sides,ydata[7],color='magenta',linestyle='--',marker='s',ms=20,label=labs[7])

   
    plt.xticks(kernel_sides,['2 x 2','3 x 3','4 x 4','5 x 5','6 x 6'],size=40)
    plt.yticks(np.arange(-15,+16,5),size=40)
    plt.grid()
    plt.legend(fontsize=25,loc=0)    
      
    if save == True:
        title = title.replace(' ','_')
        plt.savefig(title+'.png')
    if show == True:
        plt.show()
    plt.close()

def Plot_PercentDiff2 (title,ylab,labs,ydata,save=False,show=True):
    """
    Plot Metrics From a Given Dataframe
    --------------------------------
    title (str) : Title for Figure
    ylab (str) : Label for y-axis
    ydata (iter) : List of 4 arrays to plot
    --------------------------------
    Return None
    """
    plt.figure(figsize=(20,12))
    plt.title(title,size=60,weight='bold',pad=20)
    plt.ylabel(ylab,size=40,weight='bold')
    plt.xlabel('Pixel Border Depth',size=40,weight='bold')

    kernel_sides = np.array([2,3,4,5,6])
    plt.hlines(0,2,7,color='black')

    plt.plot(kernel_sides,ydata[0],color='blue',linestyle='-',marker='v',ms=20,label=labs[0])
    plt.plot(kernel_sides,ydata[1],color='green',linestyle='-',marker='v',ms=20,label=labs[1])
    plt.plot(kernel_sides,ydata[2],color='gray',linestyle='-',marker='v',ms=20,label=labs[2])
    plt.plot(kernel_sides,ydata[3],color='purple',linestyle='-',marker='v',ms=20,label=labs[3])

    plt.xticks(kernel_sides,['2 x 2','3 x 3','4 x 4','5 x 5','6 x 6'],size=40)
    plt.yticks(np.arange(-0,+11,2),size=40)
    plt.grid()
    plt.legend(fontsize=25,loc=4)    
      
    if save == True:
        title = title.replace(' ','_')
        plt.savefig(title+'.png')
    if show == True:
        plt.show()
    plt.close()

def percent_diff (a,b):
    """ Compute element-wise percent difference between arrays a and b """
    return ((a - b) / (a + b)) * 200 