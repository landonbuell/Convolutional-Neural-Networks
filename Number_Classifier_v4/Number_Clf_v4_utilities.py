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

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

            #### VARIBALE DECLARATIONS ####

N_layer_models = {
    'single_layer_models' : [(20,),(40,),(60,),(80,),(100,),(120,)] ,
    'double_layer_models' : [(20,20),(40,40),(60,60),(80,80),(100,100),
                             (120,120)],
    'quadruple_layer_models' : [(20,20,20,20),(40,40,40,40),(60,60,60,60),
                                (80,80,80,80),(100,100,100,100)] }

            #### FUNCTION DEFINTIONS ####

def train_test_data (test=0.4,seed=None):
    """ Collect Training & Testing Data from sklearn.datasets """
    print("Collecting MNIST data .....\n")
    X,y = fetch_openml(name='mnist_784',version=1,return_X_y=True)
    X_subset,y_subset = X[:16000],y[:16000]
    # return the arrays
    return train_test_split(X_subset,y_subset,
                            test_size=test,random_state=seed)

def Create_DataFrame (matrix):
    """ Create Pandas DataFrame to hold output information for each test"""
    cols = ['Train Time','Loss Value','Iterations']
    for metric in ['precision','recal']:        # prec & recal
        for bin in np.arange(0,10,1):           # classes 0 - 9
            idx = 'Class_'+str(bin)+'_'+str(metric)
            cols.append(idx)                    # add to columns list
    for I in range (0,100,1):                   # for 100 elements
        cols.append('Confmat_'+str(I))          # add 100 elements of confmat
    frame = pd.DataFrame(data=matrix,columns=cols)  # crete the frame
    return frame                                # return the frame

def Create_MLP_Model (name,layers,X,y):
    """
    Create & Train Instance of sklearn Multilayer Perceptron Class
    --------------------------------
    name (str) : name to attach to classifier instance
    test_type (str) : Indicates the type of test on that particular CSV file
    layers (tuple) : Hidden layer sizes
    X (array) : feature training data (n_samples x n_features)
    y (array) : target training data (n_samples x 1)
    --------------------------------
    Return trained MLP Classifier Instance
    """
    model = MLPClassifier(hidden_layer_sizes=layers,activation='relu',
                          solver='sgd',batch_size=100,max_iter=400,
                          tol=1e-4,random_state=None)
    setattr(model,'name',name)          # name of model
    model = model.fit(X,y)              # train model
    return model                        # return trained model

def Eval_MLP_Model(model,X,y):
    """
    Evaluate trained sklearn Multilayer Perceptron instance
    --------------------------------
    model (class) : Instance of trained MLP model
    X (array) : feature testing data (n_samples x n_features)
    y (array) : target testing data (n_samples x 1)
    --------------------------------
    return model with predictions , precision scores, & recall scores as attrbs
    """
    z = model.predict(X)                # model predictions
    labels = model.classes_             # class labels
    setattr(model,'predictions',z)
    setattr(model,'precision_scores',\
        metrics.precision_score(y,z,labels,
        average=None,zero_division=0))
    setattr(model,'recall_scores',\
        metrics.recall_score(y,z,labels,
        average=None,zero_division=0))
    return model                    # return model w/ attched attrbs

def confusion_matrix (model,y,z,show=False):
    """
    Generate Confusion Matrix for Specific Model
    --------------------------------
    model (class) : Instance of trained MLP model
    y (array) : true testing values (n_samples x 1)
    z (array) : predicted testing samples (n_samples x 1)
    show (bool) : If True, visualize color-coded confusion matrix
    --------------------------------
    Return model w/ confusion matrix (n_classes x n_classes) attrb
    """
    confmat = metrics.confusion_matrix(y,z,model.classes_)
    setattr(model,'confusion_matrix',confmat)
    if show == True:
        plt.title(str(model.name),size=20,weight='bold')
        plt.xlabel('Predicted Classes',size=16,weight='bold')
        plt.ylabel('Actual Classes',size=16,weight='bold')
        plt.imshow(confmat,cmap=plt.cm.binary)
        plt.show()
    return model





