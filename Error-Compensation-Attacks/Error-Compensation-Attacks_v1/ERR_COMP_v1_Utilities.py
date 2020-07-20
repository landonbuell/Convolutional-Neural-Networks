"""
Landon Buell
Qioyan Yu
Error-Comp-v1
6 July 2020
"""
        
            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

import tensorflow as tf
import tensorflow.keras as keras

            #### VARIABLES ####



N_layer_models = {'Single_Layer':   [(2,),(3,),(4,),(5,),(6,)],
                  'Double_Layer':   [(2,2),(3,3),(4,4),(5,5),(6,6)],
                  #'Triple_Layer':   [(2,2,2),(3,3,3),(4,4,4),(5,5,5),(6,6,6)],
                  #'Quadruple_Layer':[(2,2,2,2),(3,3,3,3),(4,4,4,4),(5,5,5,5),(6,6,6,6)]
                  }

dataframe_cols = ['Model','Average Loss','Average Precision','Average Recall','Average Train Time']

#approx_index = np.concatenate((np.arange(0,6),np.arange(26,32)),axis=-1)
approx_index = np.arange(0,2)
outfile_name = 'Comp2.csv'

output_path = 'C:\\Users\\Landon\\Documents\\GitHub\Convolutional-Neural-Networks\\' + \
                'Error-Compensation-Attacks\\Raw_Data_v1'

            #### CLASS OBJECTS ####

class ApproximationLayer (keras.layers.Layer):
    """ Approximation Layer Object, 
            Inherits from Keras Layer Object """

    def __init__(self,rows=[],cols=[],name=''):
        """ Initialize Layer Instance """
        super(ApproximationLayer,self).__init__(trainable=False,name=name)    
        self.rows = rows    # rows to approx
        self.cols = cols    # cols to approx
        self.nchs = 3       # number of channels
        self.W = self.init_W()

    def init_W (self):
        """ Create Approximation weighting Matrix """
        W = np.ones(shape=[32,32,3])
        W[self.rows[0]:self.rows[-1]] = 0          # remove top rows
        W[32-self.rows[-1]:32-self.rows[0]] = 0    # remove bottom rows
        W[:,self.rows[0]:self.rows[-1]] = 0          # remove top rows
        W[:,32-self.rows[-1]:32-self.rows[0]] = 0    # remove bottom rows
        W = tf.Variable(W,trainable=False,dtype='float32')
        return W
        
    def call (self,X):
        """ Call Layer Object w/ X, return output Y """
        now = datetime.datetime.now()
        if (now.microsecond % 2) == 0 :     # even microsecond
            # beta = True, commence attack!
            return self.W*X     # return hadamard prod
        else:                   # use exact
            # beta = False, use exact approximation
            return X            # return as-is

class CompensationLayer (keras.layers.Layer):
    """ Compensation Layer Object, 
            Inherits from Keras Layer Object """

    def __init__(self,rows=[],cols=[],name=''):
        """ Initialize Layer Instance """
        super(CompensationLayer,self).__init__(trainable=False,name=name)
      
        self.rows = rows        # rows to compensate
        self.cols = cols        # cols to compensate
        self.rowlen = len(self.rows)
        self.collen = len(self.cols)
        self.nchs = 3           # number of channels

    def compensate(self,X):
        """ Apply Compensation to samples in batch X """   
        # Top & Bottom
        X = X[:,:32-self.rows[-1],:,:]      # remove bottom
        X = X[:,self.rows[-1]:,:,:]         # remove top
        top_patch = X[:,:self.rows[-1],:,:]
        btm_patch = X[:,-self.rows[-1]:,:,:]
        X = tf.concat([top_patch,X,btm_patch],1)
        # Left & right
        X = X[:,:,:32-self.rows[-1],:]      # remove right
        X = X[:,:,self.rows[-1]:,:]         # remove left
        lft_patch = X[:,:,:self.rows[-1],:]
        rgt_patch = X[:,:,-self.rows[-1]:,:]
        X = tf.concat([lft_patch,X,rgt_patch],2)
        return X

    def call (self,X):
        """ Call Layer Object w/ X, return output Y"""
        Y = self.compensate(X)
        return Y

        #### PREPROCESSING DEFINITIONS ####

def Load_CIFAR10():
    """ Load in CFAR-10 Data set """
    print("Loading CiFAR-10 Data...\n")
    (X_train,y_train),(X_test,y_test) = keras.datasets.cifar10.load_data()
    return X_train,y_train,X_test,y_test

def Network_Model (name,kernel_sizes,rows,cols):
    """ Create Keras Neural Network Sequential Object """
    model = keras.models.Sequential(name=name)
    model.add(keras.layers.InputLayer(input_shape=(32,32,3),name='Input'))

    model.add(ApproximationLayer(rows=rows,cols=cols,name='Approx'))
    model.add(CompensationLayer(rows=rows,cols=cols,name='Comp'))

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



