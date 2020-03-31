"""
Landon Buell
Qiaoyan Yu
MNIST Tests v3 - Functions
31 March 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics

            #### FUNCTION DEFINITIONS ####  

def Train_MLP_Model (name,layers,X,y,seed=None):
    """
    Create& Train a new MLPCLassifier Instance
    --------------------------------
    name (str) : Name to attach as attribute to instance
    layers (tuple) : Sizes of hidden layers within network
    X (array) : n x m feature matrix for training - n_samples x m_features
    y (array) : n x 1 target vector for training - n_samples x 1 label
    seed (int) : Random state seed used to reproduce results
    --------------------------------
    Returns a Trained MLPClassifier Instance
    """
    MLP_CLF = MLPClassifier(hidden_layer_sizes=layers,activation='relu',
                            solver='sgd',batch_size=100,
                            max_iter=400,tol=1e-4,random_state=seed)
    setattr(MLP_CLF,'name',name)        # attatch name attribute
    MLP_CLF = MLP_CLF.fit(X,y)          # fit the target data
    return MLP_CLF                      # return classifier instance

def Confusion_Matrix (model,ytrue,ypred,show=False):
    """
    Compute Confusion Matrix for Evaluated model
    --------------------------------
    model (class) : A trained MLP Classifier instance
    ytrue (array) : 1 x N vector of ground truth values
    ypred (array) : 1 x N vector of predicted model values
    show (bool) : Create and Show figure if True (False by default)
    --------------------------------
    Return confusion matrix as array 
    """
    matrix = metrics.confusion_matrix(ytrue,ypred)
    if show == True:
        plt.title(str(model.name),size=20,weight='bold')
        plt.imshow(matrix,cmap=plt.cm.binary)
        #plt.xticks(labs)
        #plt.yticks(labs)
        plt.xlabel('Predicted Class',size=12,weight='bold')
        plt.ylabel('Actual Class',size=12,weight='bold')
        plt.show()
    return matrix

def metric_scores (ytrue,ypred):
    """
    Compute precision & recall scores for classifier model
    --------------------------------
    ytrue (array) : 1 x N vector of ground truth values
    ypred (array) : 1 x N vector of predicted model values
    --------------------------------
    Return array of precision scores followed by recall scores
    """
    precis = metrics.precision_score(ytrue,ypred,average=None)
    recall = metrics.recall_score(ytrue,ypred,average=None)
    scores = np.array([precis,recall]).ravel()
    return scores

def create_dataframe(data):
    """
    Create Pandas dataframe to hold all information pertaining to 
    --------------------------------
    data (array) : N x 22 numpy array to convert into Pandas DataFrame
    --------------------------------
    Return empty datafram w/ column labels setup
    """
    cols = ['Name','train time [s]','layers','loss val','iters']   # list to hold col names
    for metric in ['precs','recall']:               # for precision/recall scores
        for I in range (0,10,1):                    # for classes 0-9
            cols.append('Class_'+str(I)+'_'+str(metric))
    frame = pd.DataFrame(data=data,columns=cols)
    return frame                        # return the data frame


def Plot_Image (image,label,save=False,show=False):
    """
    Produce Matplotlib figure of digit w/ label
    --------------------------------
    image (array) : 1 x 784 pixel MNIST sample
    label (int) : label for corresponding MNIST sample
    save (bool) : If True, save figure to CWD
    show (bool) : if True, show figure to user
    --------------------------------
    """
    image = image.reshape(28,28)
    plt.title("Image Label: "+str(label),size=30,weight='bold')
    plt.imshow(image,cmap=plt.cm.binary,interpolation='nearest')
    if save == True:
        plt.savefig(str(label)+'.png')
    if show == True:
        plt.show()


