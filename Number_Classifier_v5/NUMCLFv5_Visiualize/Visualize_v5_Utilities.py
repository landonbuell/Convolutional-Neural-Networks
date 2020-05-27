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
        layers = ['single_layer','double_layer','triple_layer','quadruple_layer']
        idxs = [np.arange(0,6),np.arange(6,12),np.arange(12,18),np.arange(18,24)]

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

def Plot_Metrics (objs=[],attrbs='',title='',save=False,show=True):
    """
    Create MATPLOTLIB visualizations of data
    --------------------------------
    objs (iter) : List of object instances to use
    attrbs (str) : Object attribute to plot
    title (str) : title for figure
    --------------------------------
    Return None
    """
    # Initialize
    fig,ax = plt.subplots(nrows=1,ncols=4,figsize=(20,8))
    plt.suptitle(title,size=40,weight='bold')
    neurons = np.arange(20,121,20)

    # 0-th Subplot  
    data = np.array([x.__getattribute__(attrbs)['Avg_Loss'] for x in objs])
    ax[0].plot(neurons,data[0],color='blue',linestyle='-.',marker='o')
    ax[0].plot(neurons,data[1],color='orange',linestyle='--',marker='^')
    ax[0].grid(True)
    ax[0].set_yscale('log')
    ax[0].set_ylabel("Average Loss Value",size=20,weight='bold')
    ax[0].set_xlabel("Neuron Density",size=20,weight='bold')

    # 1-th Subplot
    data = np.array([x.__getattribute__(attrbs)['Avg_Iters'] for x in objs])
    ax[1].plot(neurons,data[0],color='blue',linestyle='-.',marker='o')
    ax[1].plot(neurons,data[1],color='orange',linestyle='--',marker='^')
    ax[1].grid(True)
    ax[1].set_ylabel("Average Training Iterations",size=20,weight='bold')
    ax[1].set_xlabel("Neuron Density",size=20,weight='bold')

    # 2-th Subplot
    data = np.array([x.__getattribute__(attrbs)['Avg_Prec'] for x in objs])
    ax[2].plot(neurons,data[0],color='blue',linestyle='-.',marker='o')
    ax[2].plot(neurons,data[1],color='orange',linestyle='--',marker='^')
    ax[2].grid(True)
    ax[2].set_yticks(np.arange(0,1.1,0.1))
    ax[2].set_ylabel("Average Precision Score",size=20,weight='bold')
    ax[2].set_xlabel("Neuron Density",size=20,weight='bold')

    # 3-th Subplot
    data = np.array([x.__getattribute__(attrbs)['Avg_Recall'] for x in objs])
    ax[3].plot(neurons,data[0],color='blue',linestyle='-.',marker='o')
    ax[3].plot(neurons,data[1],color='orange',linestyle='--',marker='^')
    ax[3].grid(True)
    ax[3].set_yticks(np.arange(0,1.1,0.1))
    ax[3].set_ylabel("Average Recall Score",size=20,weight='bold')
    ax[3].set_xlabel("Neuron Density",size=20,weight='bold')


    if save == True:
        plt.savefig(title+'.png')
    if show == True:
        plt.show()
    plt.close()