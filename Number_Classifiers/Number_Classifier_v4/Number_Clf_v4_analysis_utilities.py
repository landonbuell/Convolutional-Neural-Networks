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

cols_to_use = np.arange(1,24,1,dtype=int)
out_frame_idx = ['train time min','train time max','train time avg',
                 'loss value min','loss value max','loss value avg',
                 'iterations min','iterations max','iterations avg',
                 'average precision','average recall']

            #### CLASS OBJECT DEFINTIONS ####

class file_data ():
    """
    Creates object to Analyze Program output
    --------------------------------
    filename (str) : filename where data originates
    dataframe (DataFrame) : Dataframe that object will inherit data frome
    model_family (str) : Family (number of layers) for instance
    test_type (str) : Test type condiucted on instance
    nodes (int) :  number of nodes in each layer
    --------------------------------
    Return initializec class instance
    """
    def __init__(self,filename,dataframe,model_family,test_type,nodes):
        """ Initialize Class object """
        self.filename = filename
        self.family = model_family
        self.test_type = test_type
        self.n_neurons = nodes
        self.columns = dataframe.columns
        for col in dataframe.columns:
            dataset = dataframe[col].to_numpy()
            col = col.replace(' ','_')        # replace space w/ underscore
            setattr(self,str(col)+'_min',np.min(dataset))
            setattr(self,str(col)+'_max',np.max(dataset))
            setattr(self,str(col)+'_avg',np.mean(dataset))
        self.average_precision_recall(dataframe)

    def average_precision_recall(self,frame):
        """ Compute average of all recall scores """
        precs,recls = np.array([]),np.array([])

        for num in np.arange(0,10,1):                   # for each class
            idx = 'Class_'+str(num)+'_precision'        # index column name
            precs = np.append(precs,frame[idx].mean())  # add col avg to arr
        for num in np.arange(0,10,1):                   # for each class
            idx = 'Class_'+str(num)+'_recal'            # index column name (misspelled 'recall')
            recls = np.append(recls,frame[idx].mean())  # add col avg to arr

        setattr(self,'precision_avg',np.mean(precs))    # average precision across 100 samples x 10 classes
        setattr(self,'recall_avg',np.mean(recls))       # average recall across 100 samples x 10 classes

    def index_for_file (self):
        """ Return index for datframe row """
        return self.family + '_' + self.filename         # index for row of dataframe

    def data_for_file (self):
        """ Return needed data for datframe row """
        rowdata = np.array([self.Train_Time_min,self.Train_Time_max,self.Train_Time_avg,
                            self.Loss_Value_min,self.Loss_Value_max,self.Loss_Value_avg,
                            self.Iterations_min,self.Iterations_max,self.Iterations_avg,
                            self.precision_avg,self.recall_avg])
        return rowdata


        


            #### FUNCTION DEFINTIIONS ####

def Analyze_Model(model_list,test_types):
    """
    Analyze Data from specific Models in larger structrue
    --------------------------------
    model_list (list) : list instances of dataset objects for particular layer
    test_types (list) : list of strings indicating the type of tests done on data set
    --------------------------------
    Return None
    """
    N_neurons = [20,40,60,80,100,120]
    num_test = len(test_types)


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

    

    

