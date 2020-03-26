"""
Landon Buell
Qiaoyan Yu
Approximate Comp v1
24 March 2020
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

def Test_MLP_Model (model,X):
    """
    Test & Evaluate an exisiting MLPClassifier Instance
    --------------------------------
    model (class) : A trained MLP Classifier instance
    X (array) : n x m feature matrix for testing - n_samples x m_features
    y (array) : n x 1 target vector for testing - n_samples x 1 label
    --------------------------------
    Return Model and Predictions
    """
    return model.predict(X)     # run predictions

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

def precisions (confmat):
    """
    Compute Precision scores for each individual class
    --------------------------------
    confmat (array) : square confusion matrix to uss
    --------------------------------
    returns array of precision scores
    """
    scores = np.array([])               # array to hold precision scores
    for bin in range(len(confmat)):     # each row in the matrix
        pass


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

