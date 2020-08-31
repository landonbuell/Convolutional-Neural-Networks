"""
Landon Buell
Qioayan Yu
What-to-Approximate Scheme
18 August 2020
"""

        #### IMPORTS ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow.keras as keras

        #### VARIABLE DECLARATIONS ####

KERNELSIZES = np.array([2,3,4,5,6])

FrameCols = ['Model Name','Loss','Precision','Recall']
FrameName = 'Approx300.csv'

        #### FUNCTIONS DEFINITIONS ####

def CIFAR10 ():
    """ Collect CiFar 10 Data Set from tf.keras """
    print("Collecting CiFar-10 Dataset:")
    (X_train,y_train),(X_test,y_test) = keras.datasets.cifar10.load_data()
    return X_train,y_train,X_test,y_test

def PlotSample(X):
    """ Plot sample X """
    plt.figure(figsize=(16,12))
    plt.imshow(X)
    plt.show()
    

        #### CLASS DEFINTIONS ####

class WhatToApproximateLayer (keras.layers.Layer):
   """ What-To-Approximate Layer For CNN Network """
   
   def __init__(self,maskSize=100):
       """ Initialize Class Object Instance """
       super(WhatToApproximateLayer,self).__init__(trainable=False)
       self.maskSize = maskSize
       self.W = self.InitWeights()
       
   def InitWeights (self):
        """ Intialize Weight Matrix """
        W =  np.ones(shape=(32,32),dtype=np.uint8)   
        maskRows = np.random.randint(low=0,high=31,size=self.maskSize)
        maskCols = np.random.randint(low=0,high=31,size=self.maskSize)
        for i,j in zip(maskRows,maskCols):
            W[i,j] = 0
        W = np.random.permutation(W).reshape(32,32)
        W = np.transpose(np.array([W,W,W]),axes=(1,2,0))
        return W

   def Call (self,X):
       """ Call WTA layer """
       return X * self.W    # hadamard product w/ weights


class NeuralNetwork :
    """ Create Nerual Network Models """

    def __init__(self,name,kernelSizes):
        """ Initialize Class Intance """
        self.modelName = name
        self.kernelSizes = ([kernelSizes,kernelSizes],)
        self.model = self.ConvolutionalNeuralNetwork()

    def ConvolutionalNeuralNetwork (self):
        """
        Create Single Convolutional Neural Network
        --------------------------------
        name (str) : Name to Use for model instance
        kernelSizes (iter) : Iterable of ints determining wall sides of CNN kernel
        --------------------------------
        Return Compiled Keras CNN
        """
        model = keras.models.Sequential(name=self.modelName)
        model.add(keras.layers.InputLayer(input_shape=(32,32,3),name='Input'))
        model.add(WhatToApproximateLayer(maskSize=300))
        # Each Layer Group
        for i,side in enumerate(self.kernelSizes):
            model.add(keras.layers.Conv2D(filters=64,kernel_size=side,strides=(1,1),padding='same',
                                            activation='relu',name='C'+str(i+1)+'A'))
            model.add(keras.layers.Conv2D(filters=64,kernel_size=side,strides=(1,1),padding='same',
                                            activation='relu',name='C'+str(i+1)+'B'))
            model.add(keras.layers.MaxPool2D(pool_size=(2,2),name='P'+str(i+1)))

        # Dense Layers
        model.add(keras.layers.Flatten(name='F1'))
        #model.add(WhatToApproximateLayer(maskSize=100))
        model.add(keras.layers.Dense(units=128,activation='relu',name='D1'))
        model.add(keras.layers.Dense(units=10,activation='softmax',name='Output'))
 
        # Complie & Return
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=['Precision','Recall'])
        #print(model.summary())
        return model

    def __TRAIN__(self,X,Y):
        """ Train Neural Network Model """     
        self.model.fit(x=X,y=Y,batch_size=128,epochs=16,verbose=2)
        return self

    def __EVALUATE__(self,X,Y):
        """ Evaluate Neural Network Model """
        scores = self.model.evaluate(x=X,y=Y,verbose=0)
        return scores


