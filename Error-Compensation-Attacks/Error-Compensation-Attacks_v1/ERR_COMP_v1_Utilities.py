"""
Landon Buell
Qioyan Yu
Error-Comp-v1
6 July 2020
"""
        
            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
import tensorflow.keras as keras

            #### VARIABLES ####

N_layer_models = {'Single_Layer':   [(2,),(3,),(4,),(5,),(6,)],
                  #'Double_Layer':   [(2,2),(3,3),(4,4),(5,5),(6,6)],
                  #'Triple_Layer':   [(2,2,2),(3,3,3),(4,4,4),(5,5,5),(6,6,6)],
                  #'Quadruple_Layer':[(2,2,2,2),(3,3,3,3),(4,4,4,4),(5,5,5,5),(6,6,6,6)]
                  }

dataframe_cols = ['Model','Average Loss','Average Precision','Average Recall']

approx_index = np.concatenate((np.arange(0,6),np.arange(26,32)),axis=-1)
outfile_name = 'Comp_6.csv'

output_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Error-Compensation-Attacks/Raw_Data'


            #### CLASS OBJECTS ####

class ApproximationLayer (keras.layers.Layer):
    """ Approximation Layer Object, 
            Inherits from Keras Layer Object """

    def __init__(self,rows=[],cols=[]):
        """ Initialize Layer Instance """
        super(ApproximationLayer,self).__init__(trainable=False)
        
        self.rows = rows    # rows to approx
        self.cols = cols    # cols to approx
        self.nchs = 3       # number of channels

    def approximate (self,X,beta):
        """ Apply Aproximations to samples in batch X """
        W,H = X.shape[1],X.shape[2]
        X = np.copy(X)                  
        for x in X:                         # each sample    
            for r in self.rows:             # each row to approximate     
                for w in range(W):          # full width
                    for j in range(self.nchs):      # each channel (RGB?)
                        # Mute all bits
                        #x[r][w][j] -= 128 if (x[r][w][j] >= 128) else (x[r][w][j])
                        x[r][w][j] = 0
            for c in self.cols:             # each col to approximate
                for h in range(H):          # full height
                    for j in range(self.nchs):  # each channel (RGB)
                        # Mute all bits
                        #x[h][c][j] -= 128 if (x[h][c][j] >= 128) else (x[h][c][j])
                        x[h][c][j] = 0
        return X

    def call (self,X):
        """ Call Layer Object w/ X, return output Y """
        beta = True         # additional trigger condition
        Y = self.approximate(X,beta)
        return Y

class CompensationLayer (keras.layers.Layer):
    """ Compensation Layer Object, 
            Inherits from Keras Layer Object """

    def __init__(self,rows=[],cols=[]):
        """ Initialize Layer Instance """
        super(CompensationLayer,self).__init__(trainable=False)
        
        self.b = int(len(rows)/2)
        self.rows = rows        # rows to compensate
        self.toprows = rows[:self.b]
        self.botrows = rows[self.b:]
        self.cols = cols        # cols to compensate
        self.nchs = 3           # number of channels

    def compensate(self,X):
        """ Apply Compensation to samples in batch X """
        b = len(self.rows)/2         # approx border width
        X = np.copy(X)         
        for x in X:             # each sample 

            x[self.toprows] = x[self.b:2*self.b]                # patch top
            x[self.botrows] = x[(32-(2*self.b)):(32-self.b)]    # patch bottom         
            x = np.transpose(x,axes=(1,0,2))

            x[self.toprows] = x[self.b:2*self.b]                # patch top
            x[self.botrows] = x[(32-(2*self.b)):(32-self.b)]    # patch bottom 
            x = np.transpose(x,axes=(1,0,2))
                
        return X

    def call (self,X):
        """ Call Layer Object w/ X, return output Y"""
        Y = self.compensate(X)
        return Y

class AbsoluteValueLayer (keras.layers.Layer):
    """ Compute absolute value of inputs """

    def __init__(self,rows=[],cols=[]):
        """ Initialize Layer Instance """
        super(AbsoluteValueLayer,self).__init__(trainable=False)

    def call (self,X):
        """ return absolute value """
        return tf.math.abs(X)

        #### PREPROCESSING DEFINITIONS ####

def Load_CIFAR10():
    """ Load in CFAR-10 Data set """
    print("Loading CiFAR-10 Data...\n")
    (X_train,y_train),(X_test,y_test) = keras.datasets.cifar10.load_data()
    return X_train,y_train,X_test,y_test

def Network_Model (name,kernel_sizes):
    """ Create Keras Neural Network Sequential Object """
    model = keras.models.Sequential(name=name)
    model.add(keras.layers.InputLayer(input_shape=(32,32,3),name='Input'))

    # Each Layer Group
    for i,side in enumerate(kernel_sizes):
        model.add(keras.layers.Conv2D(filters=64,kernel_size=side,strides=(1,1),padding='same',
                                      activation='relu',name='C'+str(i+1)+'A'))
        model.add(keras.layers.Conv2D(filters=64,kernel_size=side,strides=(1,1),padding='same',
                                      activation='relu',name='C'+str(i+1)+'B'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),name='P'+str(i+1)))

    # Dense Layers
    model.add(keras.layers.Flatten(name='F1'))
    model.add(keras.layers.Dense(units=128,activation='relu',name='D1'))
    model.add(keras.layers.Dense(units=10,activation='softmax',name='Output'))
    

    # Complie & Return
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['Precision','Recall'])
    #print(model.summary())
    return model

        #### PLOTTING FUNCTIONS ####

def Plot_Sample (X,y,save=False,show=True):
    """ Plot RGB Image w/ Label """
    plt.figure(figsize=(16,12))
    plt.title(str(y),size=50,weight='bold')
    
    try:
        plt.imshow(X.reshape(32,32,3),cmap=plt.cm.binary)
    except:
        plt.imshow(X)
        
    plt.xticks([])
    plt.yticks([])

    if save == True:
        plt.savefig('Label_'+str(y)+'.png')
    if show == True:
        plt.show()



