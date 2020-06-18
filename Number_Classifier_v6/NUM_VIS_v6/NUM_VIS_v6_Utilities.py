"""
Landon Buell
Qioayan Yu
CLF v6 Visualize - Utilities
15 June 2020
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow.keras as keras

        #### FUNCTION DEFINITIONS ####

dataframe_columns = ['Name','Avg_Loss','Avg_Prec','Avg_Recall']

        #### CLASS OBJECT DEFINITIONS ####

class filedata ():
    """
    Create object to organize 
    """
    
    def __init__(self,name,path,filename):
        """ Initialize Class Instance"""
        self.name = name
        self.filename = filename
        self.X = pd.read_csv(path+'/'+filename,header=0,
                        usecols=dataframe_columns)
        self.densities = np.arange(20,121,20)
     
    def split_X (self):
        """ Split Frame X Based on Model Depths """
        layers = ['single_layer','double_layer']
        idxs = [np.arange(0,6),np.arange(6,12)]

        for n_layers,pts in zip(layers,idxs):
            data = self.X.loc[pts]
            setattr(self,n_layers,data)
        return self
        
    def make_arrays (self):
        """ Make Data Arrays for Plotting """
        n_layers = ['single_layer','double_layer','triple_layer','quadruple_layer']
        idxs = [np.arange(0,6),np.arange(6,12),np.arange(12,18),np.arange(18,24)]
        
        for col in self.X.columns:              
            column_data = self.X[col].to_numpy()

            for layers,pts in zip(n_layers,idxs):
                setattr(self,layers+'_'+col,column_data[pts])

        return self

class ApproximationLayer (keras.layers.Layer):
    """
    Create layer to Apply Approximate computations method to.
    --------------------------------
    rows (iter) : Array-like of rows to apply approximations to
    cols (iter) : Array-like of cols to apply approximations to
    --------------------------------
    Returns initiated Approximation Layer Instance
    """

    def __init__(self,rows=[0],cols=[0]):
        """ Initialize Approximation Layer Object """
        super(ApproximationLayer,self).__init__(trainable=False)
        self.rows = rows        # row indexs to apply MSB
        self.cols = cols        # rol index to apply MSB
        return None

    def mute_bit (self,x):
        """ Apply MSB operation to single float """
        m,e = np.frexp(x)
        e = 0 if (e > 1) else e
        x = np.ldexp(m,e)
        return x

    def Mute_MSB (self,X):
        """ Mute Most-Signfigicant bit in exponet of FP-64 """
        X_shape = X.shape               # original shape
        for i in range (len(X)):        # each samples
            for r in self.rows:         # each row   
                for j in range(0,28):   # all columns
                    X[i][r][j] = self.mute_bit(X[i][r][j])
            for c in self.cols:         # each row   
                for j in range(0,28):   # all columns
                    X[i][j][c] = self.mute_bit(X[i][j][c])
        return X                        # return new activations
        
    def call (self,inputs):
        """ Define Compution from input to produce outputs """
        output = self.Mute_MSB(np.copy(inputs))
        return output

def Load_MNIST ():
    """ Collect Training & Testing Data from keras.datasets """
    print("Collecting MNIST data .....\n")
    (X_train,y_train),(X_test,y_test) = \
        keras.datasets.mnist.load_data()
    X_test,y_test = X_test[:6000],y_test[:6000]
    X_train,y_train = X_train[:10000],y_train[:10000]
    return X_train,y_train

def Plot_Matrix (X,title='',save=False,show=False):
    """
    Visualize 2D Matrix
    --------------------------------
    X (arr) : Matrix (n_rows x n_columns) to visualize
    title (str) : title for figure
    --------------------------------
    Return None
    """
    plt.title(title,size=40,weight='bold')
    plt.imshow(X,cmap=plt.cm.binary)
    plt.xticks(np.arange(0,28,4),fontsize=20)
    plt.yticks(np.arange(0,28,4),fontsize=20)
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
    #plt.plot(neurons,data[4],color='gray',linestyle='-',marker='v',ms=16,label=labs[4])

    if metric == 'Avg_Loss':
        plt.yticks(np.arange(0,3.1,0.5))

    else:
        plt.yticks(np.arange(0,1.1,0.1))

    plt.grid()
    plt.legend(loc='upper left',fontsize=25)
    plt.yticks(size=40)
    plt.xticks(size=40)
    
    if save == True:
        title = title.replace(' ','_')
        plt.savefig(title+'.png')
    if show == True:
        plt.show()
    plt.close()
