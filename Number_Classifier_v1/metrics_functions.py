"""
Landon Buell
Prof. Yu
Metric Responses - Functions
10 January 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn.metrics as metrics
import sklearn.model_selection as model
from sklearn.linear_model import SGDClassifier


            #### FUNCTIONS DEFINITIONS ####

def SGD_CLF (name,X,y,state=None):
    """ Create and Train SGD Classifier from sklearn """
    CLF = SGDClassifier(random_state=state) # create classifier
    setattr(CLF,'name',name)                # attach name
    CLF.fit(X,y)                            # fit the dataset model
    return CLF                              # return the obj

def CLF_Predict (clf,y):
    """ Compute predicition score for each class """
    pass 

def split_train_test (nsamps,ratio):
    """
    generate a series of indicies for training & testing data
        Adapted from (Geron, 49) (Note: numpy is Psuedo-Random)
    --------------------------------
    nsamps (int) : number of sample data points
    ratio (float) : ratio of train: test data (0,1)
    --------------------------------
    return train / test indicices
    """
    shuffled = np.random.permutation(nsamps)    # permute idxs
    test_size = int(nsamps*ratio)               # test dataset size
    train = shuffled[test_size:].tolist()       # set training idxs
    test = shuffled[:test_size]                 # set testing idxs
    return train,test                           # return data pts

def write_metrics ():
    """ Write out Classifier Metrics as DataFrame for external use """
    pass

            #### METRICS ####

def confusion_matrix (CLF,ytest,ypred,labs,show=False):
    """ Produce sklearn confusion matrix for classifier predictions """
    matrix = metrics.confusion_matrix(ytest,ypred,labels=labs)
    if show == True:
        plt.title(str(CLF.name),weight='bold')
        plt.imshow(matrix,cmap=plt.cm.gray)
        plt.show()

    return matrix

def precision_score ():
    """ Compute Precision score for class n in K-Folds Classifier """
    pass

def recall_score ():
    """ Compute Recall Score for class n in K-Folds Classifier """
    pass

def F1_score ():
    """ Compute F1-score for class n in K-Folds Classifier """
    pass


            #### VISUALIZATION FUNCTIONS ####

def Plot_Image (image,label):
    """ Produce Matplotlib figure of digit w/ label"""
    image = image.reshape(28,28)
    plt.title("Image Label: "+str(label),weight='bold')
    plt.imshow(image,cmap=plt.cm.gray,interpolation='nearest')
    plt.show()


                
