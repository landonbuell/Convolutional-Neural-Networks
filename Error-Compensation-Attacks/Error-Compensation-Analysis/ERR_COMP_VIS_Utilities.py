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

N_layer_models = {'Single_Layer':   [(2,),(3,),(4,),(5,),(6,)],
                  'Double_Layer':   [(2,2),(3,3),(4,4),(5,5),(6,6)],
                  #'Triple_Layer':   [(2,2,2),(3,3,3),(4,4,4),(5,5,5),(6,6,6)],
                  #'Quadruple_Layer':[(2,2,2,2),(3,3,3,3),(4,4,4,4),(5,5,5,5),(6,6,6,6)]
                  }

class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

dataframe_cols = ['Model','Average Loss','Average Precision',
                  'Average Recall','Average Train Time']

approx_index2 = np.concatenate((np.arange(0,2),np.arange(30,32)),axis=-1)
approx_index4 = np.concatenate((np.arange(0,4),np.arange(28,32)),axis=-1)
approx_index6 = np.concatenate((np.arange(0,6),np.arange(26,32)),axis=-1)
approx_index8 = np.concatenate((np.arange(0,8),np.arange(24,32)),axis=-1)
outfile_name = ' '

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
        self.densities = np.array([2,3,4,5,6])
     
    def split_X (self):
        """ Split Frame X Based on Model Depths """
        layers = ['single_layer','double_layer']
        idxs = [np.arange(0,5),np.arange(5,10)]

        for n_layers,pts in zip(layers,idxs):
            data = self.X.loc[pts]
            data = np.round(data,4)     # round to 4 decimals
            setattr(self,n_layers,data)
        return self
        
    def make_arrays (self):
        """ Make Data Arrays for Plotting """
        n_layers = ['single_layer','double_layer']
        idxs = [np.arange(0,5),np.arange(5,10)]
        
        for col in self.X.columns:              
            column_data = self.X[col].to_numpy()

            for layers,pts in zip(n_layers,idxs):
                setattr(self,layers+'_'+col,column_data[pts])

        return self

def Load_CIFAR10():
    """ Load in CFAR-10 Data set """
    print("Loading CiFAR-10 Data...\n")
    (X_train,y_train),(X_test,y_test) = keras.datasets.cifar10.load_data()
    return X_train,y_train,X_test,y_test

def Plot_Matrix (X,title='',save=False,show=False):
    """
    Visualize 2D Matrix
    --------------------------------
    X (arr) : Matrix (n_rows x n_columns) to visualize
    title (str) : title for figure
    --------------------------------
    Return None
    """
    #plt.title(title,size=40,weight='bold')
    plt.imshow(X)
    plt.xticks([])
    plt.yticks([])
    if save == True:
        title = title.replace(': ','_')
        plt.savefig(title.replace(' ','_')+'.png')
    if show == True:
        plt.show()

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
    plt.figure(figsize=(20,12))
    plt.title(title,size=60,weight='bold',pad=20)
    plt.ylabel(ylab,size=50,weight='bold')
    plt.xlabel('2D Kernel Shape',size=50,weight='bold')

    kernel_sides = np.array([2,3,4,5,6])
    data = np.array([x.__getattribute__(attrbs)[metric] for x in objs])
    plt.plot(kernel_sides,data[0],color='red',linestyle='-',marker='o',ms=16,label=labs[0])

    plt.plot(kernel_sides,data[1],color='blue',linestyle='--',marker='^',ms=16,label=labs[1])
    plt.plot(kernel_sides,data[2],color='cyan',linestyle='--',marker='^',ms=16,label=labs[2])
    plt.plot(kernel_sides,data[3],color='green',linestyle='--',marker='^',ms=16,label=labs[3])
    plt.plot(kernel_sides,data[4],color='purple',linestyle='--',marker='^',ms=16,label=labs[4])

    plt.plot(kernel_sides,data[5],color='gray',linestyle='-.',marker='s',ms=16,label=labs[5])
    plt.plot(kernel_sides,data[6],color='orange',linestyle='-.',marker='s',ms=16,label=labs[6])
    plt.plot(kernel_sides,data[7],color='magenta',linestyle='-.',marker='s',ms=16,label=labs[7])
    plt.plot(kernel_sides,data[8],color='yellow',linestyle='-.',marker='s',ms=16,label=labs[8])

    plt.xticks(kernel_sides,['2x2','3x3','4x4','5x5','6x6'],size=50)
    if metric in ['Average Precision','Average Recall']:
        plt.yticks(np.arange(0.2,1.0,0.1),size=50)
    else:
        plt.yticks(size=50)

    plt.grid()
    plt.legend(fontsize=25,loc=0)    
      
    if save == True:
        title = title.replace(' ','_')
        plt.savefig(title+'.png')
    if show == True:
        plt.show()
    plt.close()