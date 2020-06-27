"""
Landon Buell
Qioyan Yu
Error-Comp-v0
22 June 2020
"""
        
            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
import tensorflow.keras as keras

            #### VARIABLES ####

N_layer_models = {'Single_Layer':[(2,),(3,),(4,),(5,),],
                  'Double_Layer':[(2,2),(3,3),(4,4),(5,5),]
                  }

dataframe_cols = ['Model','Average Loss','Average Precision','Average Recall']

approx_index = np.concatenate((np.arange(0,8),np.arange(20,28)),axis=-1)
outfile_name = 'Baseline.csv'

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
        self.nchs = 1       # number of channels

    def approximate (self,X):
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
        """ Call Layer Object w/ X, return output Y"""
        Y = self.approximate(X)
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
        self.nchs = 1           # number of channels

    def compensate(self,X):
        """ Apply Compensation to samples in batch X """
        b = len(self.rows)/2         # approx border width
        X = np.copy(X)         
        for x in X:             # each sample 

            x[self.toprows] = x[self.b:2*self.b]                # patch top
            x[self.botrows] = x[(28-(2*self.b)):(28-self.b)]    # patch bottom         
            x = np.transpose(x,axes=(1,0,2))

            x[self.toprows] = x[self.b:2*self.b]                # patch top
            x[self.botrows] = x[(28-(2*self.b)):(28-self.b)]    # patch bottom 
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
    """ Create Keras MLP Object """
    model = keras.models.Sequential(name=name)
    model.add(keras.layers.InputLayer(input_shape=(32,32,3),
                                      name='Input'))
    # Add Convolutional & Pooling Layers
    for i,side in enumerate(kernel_sizes):   # Each kernel size
        model.add(keras.layers.Conv2D(filters=16,kernel_size=side,strides=(2,2),
                                      activation='relu',name='C'+str(i+1)))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2),name='P'+str(i+1)))
    #model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),name='C'+str(i+2)))
    # Add Dense Layers
    model.add(keras.layers.Flatten(name='F1'))
    model.add(keras.layers.Dense(units=64,activation='relu',name='D1'))
    model.add(keras.layers.Dense(units=10,activation='relu',name='D2'))
    model.add(keras.layers.Activation(activation='softmax',name='Output'))
    # Post-Processing
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalCrossentropy() ,
                 metrics=['Precision','Recall'])
    print(model.summary())
    return model

        #### METRIC DEFINITIONS ####

def Evaluate_Model (model,X,y):
    """ Evaluate Performance of Trained Model """
    z = model.predict(X)
    n_samples = X.shape[0]
    metrics = np.array([])
    # Loss Function Value
    y = keras.utils.to_categorical(y,10)
    loss = keras.losses.categorical_crossentropy(y,z)
    assert loss.shape == (n_samples,)
    metrics = np.append(metrics,np.mean(loss.numpy()))
    # Precision Score
    prec_inst = keras.metrics.Precision()
    prec_inst.update_state(y,z)
    prec_score = np.mean(prec_inst.result().numpy())
    metrics = np.append(metrics,prec_score)
    # Recall Score
    recall_inst = keras.metrics.Recall()
    recall_inst.update_state(y,z)
    recall_score = np.mean(recall_inst.result().numpy())
    metrics = np.append(metrics,recall_score)
    
    return metrics

        #### PLOTTING FUNCTIONS ####

def Plot_Sample (X,y,save=False,show=True):
    """ Plot RGB Image w/ Label """
    plt.figure(figsize=(16,12))
    plt.title(str(y),size=50,weight='bold')
    
    try:
        plt.imshow(X.reshape(28,28),cmap=plt.cm.binary)
    except:
        plt.imshow(X)
        

    plt.xticks([])
    plt.yticks([])

    if save == True:
        plt.savefig('Label_'+str(y)+'.png')
    if show == True:
        plt.show()


