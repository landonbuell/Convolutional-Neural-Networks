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

N_layer_models = {'Single_Layer':[(20,),(40,),(60,),(80,),(100,),(120,)],
                  'Double_Layer':[(20,20),(40,40),(60,60),(80,80),
                                  (100,100),(120,120)]}

class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

dataframe_cols = ['Model','Average Loss','Average Precision','Average Recall']

approx_index = np.concatenate((np.arange(0,10),np.arange(22,32)),axis=-1)
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
        self.nchs = 3       # number of channels

    def approximate (self,X):
        """ Apply Aproximations to samples in batch X """
        W,H = X.shape[1],X.shape[2]
        for x in X:                         # each sample    
            for r in self.rows:             # each row to approximate     
                for w in range(W):          # full width
                    for j in range(self.nchs):      # each channel (RGB)
                        # Mute MSB
                        x[r][w][j] -= 128 if (x[r][w][j] >= 128) else (x[r][w][j])
                        #x[r][w][j] = 0
            for c in self.cols:             # each col to approximate
                for h in range(H):          # full height
                    for j in range(self.nchs):  # each channel (RGB)
                        # Mute MSB
                        x[h][c][j] -= 128 if (x[h][c][j] >= 128) else (x[h][c][j])
                        #x[h][c][j] = 0
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
        super(ApproximationLayer,self).__init__(trainable=False)
        
        self.rows = rows    # rows to compensate
        self.cols = cols    # cols to compensate
        self.nchs = 3       # number of channels

    def compensate(self,X):
        """ Apply Compensation to samples in batch X """
        
        # Top-Center
        comp_rows = self.rows + len(self.rows)/2


    def call (self,X):
        """ Call Layer Object w/ X, return output Y"""
        Y = X
        return Y

        #### PREPROCESSING DEFINITIONS ####

def Load_CIFAR10(train_size=10000,test_size=6000):
    """ Load in CFAR-10 Data set """
    print("Loading CiFAR-10 Data...\n")
    (X_train,y_train),(X_test,y_test) = keras.datasets.cifar10.load_data()
    X_train,y_train = X_train[:train_size],y_train[:train_size]              
    X_test,y_test = X_test[:test_size],y_test[:test_size]
    return X_train,y_train,X_test,y_test

def Load_MNIST10(test_size=10000,train_size=6000):
    """ Load in CFAR-10 Data set """
    print("Loading CiFAR-10 Data...\n")
    (X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
    X_train,y_train = X_train[:train_size],y_train[:train_size]              
    X_test,y_test = X_test[:test_size],y_test[:test_size]
    return X_train,y_train,X_test,y_test

def Network_Model (name,layers,rows,cols):
    """ Create Keras MLP Object """
    model = keras.models.Sequential(name=name)
    model.add(keras.layers.InputLayer(input_shape=(32,32,3),
                                      name = 'Input'))
    model.add(keras.layers.Conv2D(filters=1,kernel_size=(4,4),activation='relu',
                                  kernel_initializer='ones',name='C1'))
    model.add(keras.layers.Conv2D(filters=1,kernel_size=(2,2),activation='relu',
                                  kernel_initializer='ones',name='C2'))
    model.add(keras.layers.AveragePooling2D(pool_size=(4,4),strides=(2,2),
                                            name='P1'))
    model.add(keras.layers.Flatten(name='F1'))
    for i,nodes in enumerate(layers):
        model.add(keras.layers.Dense(units=nodes,
            activation='relu',name='D'+str(i+1)))
    model.add(keras.layers.Dense(units=10,activation='softmax',
                                 name='Output'))
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['Precision','Recall'])
    #print(model.summary())
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
    plt.title('Label: '+str(y),size=50,weight='bold')
    
    try:
        plt.imshow(X)
    except:
        plt.imshow(X.reshape(32,32,3))

    plt.xticks([])
    plt.yticks([])

    if save == True:
        plt.savefig('Label_'+str(y)+'.png')
    if show == True:
        plt.show()


