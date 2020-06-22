"""
Landon Buell
Qioyan Yu
Error-Comp-v0
22 June 2020
"""
        
            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import time

import tensorflow.keras as keras

import ERR_COMP_Utilities as utils

            
            #### MAIN EXECUABLE ####

if __name__ == '__main__':
        
        # PRE-PROCESSING 
        X_train,y_train,X_test,y_test = utils.Load_CIFAR10()
        LAYER_MODELS = utils.N_layer_models
        y_train = keras.utils.to_categorical(y_train,10) 

        N_iters = 10            # Time to repeat each model
        n_epochs = 1           # epochs over data set
        output_dataframe = pd.DataFrame()   # Empty Dataframe

        # ITERATE BY LAYER
        for N_LAYERS in LAYER_MODELS.keys():            # Each number of layers
            print('Layers: '+str(N_LAYERS)+'-',time.perf_counter())     # indicate layers

            for N_NEURONS in LAYER_MODELS[N_LAYERS]:    # Each number of nodes
                print('\tDensity:'+str(N_NEURONS[0])+'-',time.perf_counter())  #inicate nodes
                model_name = str(N_LAYERS)+'_'+str()
                like_model_data = np.array([])          # hold dat from each iter

                for i in range(N_iters):            # Each Iteration
                    print('\t\tIteration'+str(i)+'-',time.perf_counter())

                    MODEL = utils.Network_Model(name=model_name,
                        layers=N_NEURONS,rows=utils.approx_index,cols=utils.approx_index)
                    HIST = MODEL.fit(x=X_train,y=y_train,batch_size=128,
                                     epochs=n_epochs,verbose=2)         # train model
                    metrics = utils.Evaluate_Model(MODEL,X_test,y_test) # evaluate performance


                # Compute Averages of N_iters Models
                like_model_data = like_model_data.reshape(N_iters,-1)
