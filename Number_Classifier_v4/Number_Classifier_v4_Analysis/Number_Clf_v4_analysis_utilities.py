"""
Landon Buell
Qiaoyan Yu
Number Classifier v4 - Analysis tilities
3 April 2020
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

            #### VARIBALE DECLARATIONS ####

N_layer_models = {
    'single_layer_models' : [(20,),(40,),(60,),(80,),(100,),(120,)] ,
    'double_layer_models' : [(20,20),(40,40),(60,60),(80,80),(100,100),
                             (120,120)],
    #'quadruple_layer_models' : [(20,20,20,20),(40,40,40,40),(60,60,60,60),
     #                           (80,80,80,80),(100,100,100,100)] 
    }

            #### CLASS OBJECT DEFINTIONS ####

class dataset ():
    """
    Creates object to Analyze Program output
    --------------------------------
    filename (str) : filename where data originates
    dataframe (DataFrame) : Dataframe
    --------------------------------
    Return initializec class instance
    """
    def __init__(self,filename,dataframe,layers):
        """ Initialize Class object """
        self.filename = filename
        self.layer_sizes = layers
        self.columns = dataframe.columns
        for col in dataframe.columns:
            dataset = dataframe[col].to_numpy()
            setattr(self,str(col)+'_min',np.min(dataset))
            setattr(self,str(col)+'_max',np.max(dataset))
            setattr(self,str(col)+'_avg',np.mean(dataset))

            #### FUNCTION DEFINTIIONS ####

def Analyze_Models(model_list,model_dict):
    """
    Analyze Data from specific Models in larger structrue
    --------------------------------
    model_list (list) : list of string corresponding to specific keys in 'model_dictionary'
    model_dict (dict) : dictinary of where vals are instances of dat for specific models
    --------------------------------
    Return None
    """
    pass

            #### VISUALIZING FUNCTIONS ####
           
def Plot_Models (models,attrbs,ylab,title,save=False,show=True):
    """
    Create visualiztions of attributes

    """
    plt.figure(figsize=(20,12))
    plt.title(str(title),size=40,weight='bold')
    plt.xlabel('Layer Sizes',size=20,weight='bold')
    plt.ylabel(ylab,size=20,weight='bold')

    plt.tight_layout()
    plt.grid()
    plt.legend()
    if save == True:
        plt.savefig(title+'.png')
    if show == true:
        plt.show()

    

    

