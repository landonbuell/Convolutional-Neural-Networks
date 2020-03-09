"""
Landon Buell
Prof. Yu
MLP Classifier
1 March 2020
"""

            #### IMPORTS ####

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

import numpy as np
import matplotlib.pyplot as plt

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    print("Loading Data:")

    # load in data set - MNIST
    MNIST = fetch_openml(name='mnist_784',version=1)
    X,y = MNIST['data'],MNIST['target']
    X,y = X[:10000],y[:10000]

    print("Data Loaded.")

    # create instance of MLP Classifier object
    layers = (20,20,20,20)
    MLP_CLF = MLPClassifier(hidden_layer_sizes=layers,activation='relu',
                            solver='sgd',max_iter=200,tol=1e-4,
                            random_state=0)

    print("Training Network:")

    # split the data into training a testing sets
    Xtrain,Xtest,ytrain,ytest = \
        train_test_split(X,y,test_size=0.4,random_state=42)

    # Train the Classifier Instance
    MLP_CLF.fit(Xtrain,ytrain)

    # Test & Evaluate Classifer

    print("Testing Network")

    ypred = MLP_CLF.predict(Xtest)

    confmat = confusion_matrix(ytest,ypred)
    plt.imshow(confmat,cmap=plt.cm.binary)
    plt.title("MNIST Confusion Matrix",size=20,weight='bold')
    plt.xticks(np.arange(0,10,1))
    plt.yticks(np.arange(0,10,1))
    plt.show()