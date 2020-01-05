"""
Landon Buell
Number Classifier v0
Functions 
27 December 2019
"""

        #### IMPORTS ####
    
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.linear_model import SGDClassifier
import sklearn.model_selection as model
import sklearn.metrics as metrics

        #### FUNCTION DEFINITIONS ####

def SGD_Classifier (xdata,ydata,max_iter,seed=0,size=0.1):
    """
    Create a Stoichastic Gradient Descent Classifier Object w/ sklearn
    ----------------
    xdata (array/DataFrame) : base dataset
    ydata (array/DataFrame) : target labels for dataset
    max_iter (int) : Maximum iterations for SGD operations
    seed (int) : Random state seed for shuffeling data (0 by default)
    size (float) : Relative size of testing data set (0.1 by default)
    ----------------
    Returns classifier object and dictionary of training/testing data
    """
    X_train,X_test = model.train_test_split(xdata,test_size=size,random_state=seed)
    Y_train,Y_test = model.train_test_split(ydata,test_size=size,random_state=seed)

    xy_dict =   {'X_train':X_train,'X_test':X_test,
                 'Y_train':Y_train,'Y_test':Y_test}     # train/test data into dictionary

    CLF = SGDClassifier(random_state=seed)    # create classifer object
    CLF.fit(X_train,Y_train)                # fit dataset   
    return CLF,xy_dict                      # return classifier & xy data dictionary

def confusion (clf,xdata,ydata):
    """
    Build Confusion matric and dictionary for K-Class Classifier
    ----------------
    clf (classifier obj) : Classifier object to build confusion matrix for
    xdata (array/DataFrame) : x-training dataset
    ydata (array/DataFrame) : y-training target dataset
    ----------------
    Returns Binary confusion matrix and dictionary of entries
    """
    ypred = model.cross_val_predict(clf,xdata,ydata)    # cross-val prediction
    conf_mat = metrics.confusion_matrix(ydata,ypred)    # build confusion matrix
    return conf_mat

def general_metrics (clf,xdata,ydata,disp=True):
    """
    Build Confusion matric and dictionary for binary classifier
    ----------------
    clf (classifier obj) : Classifier object to build confusion matrix for
    xdata (array/DataFrame) : x-training dataset
    ydata (array/DataFrame) : y-training target dataset
    disp (bool) : Display outputs to command line (True by default)
    ----------------
    Returns Binary confusion matrix and dictionary of entries
    """
    ypred = model.cross_val_predict(clf,xdata,ydata)    # cross-val prediction
    
    precision = metrics.precision_score(ydata,ypred)    # compute precision score
    recall = metrics.recall_score(ydata,ypred)          # compute recall score
    f1 = metrics.f1_score(ydata,ypred)                  # compute f1 score

    if disp == True:                        # print output to line?
        print('Precision Score:',precision)
        print('Recall Score:',recall)
        print("F1 Score:",f1)

    return precision,recall,f1                  # return values

