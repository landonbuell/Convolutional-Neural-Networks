"""
Landon Buell
Qioyan Yu
Error-Comp-Visualization-v0
24 June 2020
"""
        
            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import tensorflow as tf
import tensorflow.keras as keras

            #### VARIABLES ####

N_layer_models = {'Single_Layer':[(20,),(40,),(60,),(80,),(100,),(120,),],
                  'Double_Layer':[(20,20),(40,40),(60,60),(80,80),(100,100),(120,120),]
                  }

class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

dataframe_cols = ['Model','Average Loss','Average Precision','Average Recall']

approx_index = np.concatenate((np.arange(0,10),np.arange(18,28)),axis=-1)
outfile_name = 'Approx_10.csv'

output_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Error-Compensation-Attacks/Raw_Data'


            #### CLASS OBJECTS ####

class filedata ():
    """
    Create object to organize 
    """
    
    def __init__(self,name,path,filename):
        """ Initialize Class Instance"""
        self.name = name
        self.filename = filename
        self.X = pd.read_csv(path+'/'+filename,header=0,
                        usecols=dataframe_cols)
        self.densities = np.arange(20,121,20)
     
    def split_X (self):
        """ Split Frame X Based on Model Depths """
        layers = ['single_layer','double_layer']
        idxs = [np.arange(0,6),np.arange(6,12)]

        for n_layers,pts in zip(layers,idxs):
            data = self.X.loc[pts]
            data = np.round(data,4)     # round to 4 decimals
            setattr(self,n_layers,data)
        return self
        
    def make_arrays (self):
        """ Make Data Arrays for Plotting """
        n_layers = ['single_layer','double_layer',]
        idxs = [np.arange(0,6),np.arange(6,12),]
        
        for col in self.X.columns:              
            column_data = self.X[col].to_numpy()

            for layers,pts in zip(n_layers,idxs):
                setattr(self,layers+'_'+col,column_data[pts])

        return self

class ApproximationLayer (keras.layers.Layer):
    """ Approximation Layer Object, 
            Inherits from Keras Layer Object """

    def __init__(self,rows=[],cols=[]):
        """ Initialize Layer Instance """
        super(ApproximationLayer,self).__init__(trainable=False)
        
        self.rows = rows    # rows to approx
        self.cols = cols    # cols to approx
        self.nchs = 1       # number of channels

    def approximate (self,X):
        """ Apply Aproximations to samples in batch X """
        W,H = X.shape[1],X.shape[2]
        X = np.copy(X)                  
        for x in X:                         # each sample    
            for r in self.rows:             # each row to approximate     
                for w in range(W):          # full width
                    for j in range(self.nchs):      # each channel (RGB?)
                        # Mute all bits
                        x[r][w][j] -= 128 if (x[r][w][j] >= 128) else (x[r][w][j])
                        #x[r][w][j] = 0
            for c in self.cols:             # each col to approximate
                for h in range(H):          # full height
                    for j in range(self.nchs):  # each channel (RGB)
                        # Mute all bits
                        x[h][c][j] -= 128 if (x[h][c][j] >= 128) else (x[h][c][j])
                        #x[h][c][j] = 0
        return X

    def call (self,X):
        """ Call Layer Object w/ X, return output Y"""
        Y = self.approximate(X)
        return Y

class CompensationLayer (keras.layers.Layer):
    """ Compensation Layer Object, 
            Inherits from Keras Layer Object """

    def __init__(self,rows=[],cols=[]):
        """ Initialize Layer Instance """
        super(CompensationLayer,self).__init__(trainable=False)
        
        self.rows = rows    # rows to compensate
        self.cols = cols    # cols to compensate
        self.nchs = 1       # number of channels

    def compensate(self,X):
        """ Apply Compensation to samples in batch X """
        
        # Top-Center
        comp_rows = self.rows + len(self.rows)/2


    def call (self,X):
        """ Call Layer Object w/ X, return output Y"""
        Y = X
        return Y

def Plot_Metric (objs=[],attrbs='',metric='',ylab='',labs=[],title='',save=False,show=False):
    """
    Create MATPLOTLIB visualizations of data
    --------------------------------
    objs (iter) : List of object instances to use
    metric (str) : Classificication metric to use - ['Avg_Loss','Avg_Prec','Avg_Recall']
    ylab (str) : y-axis label for plot
    labs (iter) : labels for each sample 
    title (str) : title for figure
    --------------------------------
    Return None
    """
    plt.figure(figsize=(16,12))
    plt.title(title,size=60,weight='bold',pad=20)
    plt.ylabel(ylab,size=50,weight='bold')
    plt.xlabel('Neuron Density',size=50,weight='bold')

    neurons = np.arange(20,121,20)
    data = np.array([x.__getattribute__(attrbs)[metric] for x in objs])
    plt.plot(neurons,data[0],color='red',linestyle='-',marker='o',ms=16,label=labs[0])
    plt.plot(neurons,data[1],color='blue',linestyle='--',marker='^',ms=16,label=labs[1])
    plt.plot(neurons,data[2],color='green',linestyle='-.',marker='H',ms=16,label=labs[2])
    plt.plot(neurons,data[3],color='purple',linestyle=':',marker='s',ms=16,label=labs[3])
    plt.plot(neurons,data[4],color='gray',linestyle='-',marker='v',ms=16,label=labs[4])
    plt.plot(neurons,data[5],color='orange',linestyle='-',marker='o',ms=16,label=labs[5])

    if metric == 'Average Loss':
        plt.yticks(np.arange(0,3.1,0.5))

    else:
        plt.yticks(np.arange(0,1.1,0.1))

    plt.grid()
    plt.legend(fontsize=25)
    plt.yticks(size=40)
    plt.xticks(size=40)
    
    if save == True:
        title = title.replace(' ','_')
        plt.savefig(title+'.png')
    if show == True:
        plt.show()
    plt.close()