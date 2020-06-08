"""
Landon Buell
Number Classifier v6
Main Script
7 June 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import bitstring

import tensorflow as tf
import tensorflow.keras as keras

            #### VARIABLE DECLARATIONS #####

N_layer_models = {
    'single_layer' : [(20,),(40,),(60,),(80,),(100,),(120,)] ,
    'double_layer' : [(20,20),(40,40),(60,60),(80,80),(100,100),
                             (120,120)],
    }


            #### FUNCTION DEFINITIONS ####

def Load_MNIST ():
    """ Collect Training & Testing Data from keras.datasets """
    print("Collecting MNIST data .....\n")
    (X_train,y_train),(X_test,y_test) = \
        keras.datasets.mnist.load_data()
    X_test,y_test = X_test[:6000],y_test[:6000]
    X_train,y_train = X_train[:10000],y_train[:10000]
    X_train,X_test = X_train.reshape(10000,784),X_test.reshape(6000,784)
    return X_train,X_test,y_train,y_test

def Mute_MSB (X):
    """ Mute Most-Signfigicant bit in exponet of FP-64 """
    X_shape = X.shape                       # original shape
    mants,exps = np.frexp(X)                # mantissa,exponent
    exps = np.array([0 if (e > 0) else e for e in exps.ravel()])
    X = np.ldexp(mants,exps.reshape(X_shape)) # reconstruct FP-64
    return X                                  # return new activations

def Keras_Model (layers):
    """
    Create Keras Sequential Model
    --------------------------------
    layers (tup) : Iterable with i-th elem is units in i-th Dense Layer
    --------------------------------
    Return untrained, Compiled Keras Model
    """
    model = keras.models.Sequential(name='Digit_Classifier')
    model.add(keras.layers.Input(shape=(784),name='Image'))
    #model.add(keras.layers.Lambda(function=Mute_MSB,output_shape=(784)))
    model.add(keras.layers.Dense(units=10,activation='softmax'),name='Output')
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['Precision','Recall'])
    print(model.summary())
    return model

              #### PLOTTING FUNCTIONS ####

def Plot_Confusion(model,show=True):
    """
    Create 2D visualization of Confusion Matrix
    --------------------------------
    mode (arr) Confusion Matrix (n_classes x n_classes)
    --------------------------------
    return None
    """
    n_classes = model.confusion.shape[0]
    plt.imshow(model.confusion,cmap=plt.cm.binary)
    plt.title(model.name,size=40,weight='bold')
    plt.xlabel("Predicted Class",size=20,weight='bold')
    plt.ylabel("Actual Class",size=20,weight='bold')
    plt.xticks(np.arange(0,n_classes,1))
    plt.yticks(np.arange(0,n_classes,1))  
    if show == True:
        plt.show()

def Plot_Matrix (X,label=''):
    """
    Visualize 2D Matrix
    --------------------------------
    X (arr) : Matrix (n_rows x n_columns) to visualize
    --------------------------------
    Return None
    """
    plt.title(label,size=40,weight='bold')
    plt.imshow(X,cmap=plt.cm.binary)
    plt.show()

def Plot_History (hist,model,save=False,show=False):
    """
    Visualize Data from Keras History Object Instance
    --------------------------------
    hist (inst) : Keras history object
    --------------------------------
    Return None
    """
    # Initialize Figure

    eps = np.array(hist.epoch)          # arr of epochs
    n_figs = len(hist.history.keys())

    fig,axs = plt.subplots(nrows=n_figs,ncols=1,sharex=True,figsize=(20,8))
    plt.suptitle(model.name+' History',size=50,weight='bold')
    hist_dict = hist.history
    
    for I in range (n_figs):                # over each parameter
        key = list(hist_dict)[I]
        axs[I].set_ylabel(str(key).upper(),size=20,weight='bold')
        axs[I].plot(eps,hist_dict[key])     # plot key
        axs[I].grid()                       # add grid

    plt.xlabel("Epochs",size=20,weight='bold')

    if save == True:
        plt.savefig(title.replace(' ','_')+'.png')
    if show == True:
        plt.show()