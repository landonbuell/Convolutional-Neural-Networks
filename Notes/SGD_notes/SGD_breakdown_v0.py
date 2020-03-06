"""
Landon Buell
Prof. Yu
SGD Classifier Breakdown
18 Feb 2020
"""

            #### IMPORTS ####

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # load in data set - IRIS
    IRIS = load_iris()
    X,y = IRIS['data'],IRIS['target']

    # create instance of SGD Classifier object
    sgd_clf = SGDClassifier(random_state=0,max_iter=1000,tol=1e-3)

    # split the data into training a testing sets
    Xtrain,Xtest,ytrain,ytest = \
        train_test_split(X,y,test_size=0.1,random_state=42)
   
    # train the classifier on the data set
    sgd_clf.fit(Xtrain,ytrain)
