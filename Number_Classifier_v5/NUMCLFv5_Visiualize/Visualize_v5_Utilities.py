"""
Landon Buell
Qiaoyan Yu
Number Classifier v4 - Utilities
3 April 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


            #### VARIBALE DECLARATIONS ####

N_layer_models = {
    'single_layer' : [(20,),(40,),(60,),(80,),(100,),(120,)] ,
    'double_layer' : [(20,20),(40,40),(60,60),(80,80),(100,100),
                             (120,120)],
    'triple_layer' : [(20,20,20),(40,40,40),(60,60,60),(80,80,80),
                      (100,100,100),(120,120,120)],
    'quadruple_layer' : [(20,20,20,20),(40,40,40,40),(60,60,60,60),
                            (80,80,80,80),(100,100,100,100),(120,120,120,120)] 
    }

dataframe_columns = ['Name','Avg_Loss','Avg_Iters','Avg_Prec','Avg_Recall']


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

    if metric == 'Avg_Loss':
           plt.yscale('log')

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
