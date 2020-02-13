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

def SGD_CLF (name,xtrain,ytrain,xtest,state=None):
    """ Create, Train & EvaluateSGD Classifier from sklearn """
    CLF = SGDClassifier(random_state=state,
                        max_iter=1000,tol=1e-3) 
    setattr(CLF,'name',name)                # attach name
    CLF.fit(xtrain,ytrain)                  # fit the dataset model
    preditions = CLF.predict(xtest)         # predict new data
    return CLF,preditions                   # return obj & outputs

def add_noise(data,mean,stddev,type='gaussian'):
    """ Add pseudo-random noise to arrays """
    org_shape = np.shape(data)      # original shape of the array
    data = data.flatten()           # flatten the array
    if type == 'gaussian':
        noise = np.random.normal(mean,stddev,len(data))
    data += noise                   # add noise in
    return data.reshape(org_shape)  # reshape & return

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
    matrix = metrics.confusion_matrix(ytest,ypred)
    if show == True:
        plt.title(str(CLF.name),size=20,weight='bold')
        plt.imshow(matrix,cmap=plt.cm.binary)
        plt.xticks(labs)
        plt.yticks(labs)
        plt.xlabel('Actual Class',size=12,weight='bold')
        plt.ylabel('Predicted Class',size=12,weight='bold')
        plt.show()
    return matrix

def cross_validation (clf,xtrain,ytrain,k):
    """ Impliment Cross - Validation Prediction algorithm """
    pred = model.cross_val_predict(clf,X=xtrain,y=ytrain,cv=k)
    # WARNING : THIS METHOD TAKES A LONG TIME TO RUN!
    return pred

def classification_scores (ytrue,ypred,labs,avg='macro'):
    """ 
    Compute Precision, Recall & F1 Scores given classsifier predictions 
    --------------------------------
    ytrue (array) : training labels for test data set (1 x M)
    ypred (array) : training predictions for test data set (1 x M)
    labs (list) : multiclass labels used for classifier
    avg (str) : average key to use for socre (see sklearn documentation)
    --------------------------------
    Return dictionary of classification scores
    """
    precision = metrics.precision_score(ytrue,ypred,labels=labs,average=avg)
    recall = metrics.recall_score(ytrue,ypred,labels=labs,average=avg)
    f1_sc = metrics.f1_score(ytrue,ypred,labels=labs,average=avg)
    scores = np.array([precision,recall,f1_sc])
    return scores                   # return the array


            #### VISUALIZATION FUNCTIONS ####

def Plot_Image (image,label,save=False,show=False):
    """ Produce Matplotlib figure of digit w/ label"""
    image = image.reshape(28,28)
    plt.title("Image Label: "+str(label),size=30,weight='bold')
    plt.imshow(image,cmap=plt.cm.binary,interpolation='nearest')
    if save == True:
        plt.savefig(str(label)+'.png')
    if show == True:
        plt.show()


                
