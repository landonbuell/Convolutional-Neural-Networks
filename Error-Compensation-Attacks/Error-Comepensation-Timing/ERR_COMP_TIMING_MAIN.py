"""
Landon Buell
Qioyan Yu
Error-Comp-Timing-Utilities
29 July 2020
"""
        
            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

import tensorflow as tf
import tensorflow.keras as keras

import ERR_COMP_TIMING_Utils as utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # PRE-PROCESSING 
    init_path = os.getcwd()
    X_train,y_train,X_test,y_test = utils.Load_CIFAR10()
    
    y_train = keras.utils.to_categorical(y_train,10) 
    y_test = keras.utils.to_categorical(y_test,10)
    X_test = X_test[:1280]

    LAYER_MODELS = utils.N_layer_models

    # DIRECTORY
    output_frame = pd.DataFrame(columns=utils.dataframe_cols)   # create dataframe
    outpath = os.path.join(utils.output_path,utils.outfile_name)
    if os.path.isfile(outpath):     # file exists
        pass                        # do nothing
    else:                           # does not exist
        output_frame.to_csv(outpath,
                    columns=utils.dataframe_cols,mode='w')      # make new file

    N_iters = 4             # Time to repeat each model
    n_epochs = 4            # epochs over data set

    # ITERATE BY LAYER
    for N_LAYERS in LAYER_MODELS.keys():            # Each number of layers
        print('Layers: '+str(N_LAYERS)+'-',time.perf_counter())     # indicate layers

        for KERNEL_SIZE in LAYER_MODELS[N_LAYERS]:    # Each number of nodes
            print('\tDensity: '+str(KERNEL_SIZE[0])+'-',time.perf_counter())  #inicate nodes
            model_name = str(N_LAYERS)+'_'+str(KERNEL_SIZE[0])
            like_model_data = np.array([])          # hold dat from each iter
                
            for i in range(N_iters):            # Each Iteration
                print('\t\tIteration: '+str(i)+'-',time.perf_counter())

                MODEL = utils.Network_Model(name=model_name,kernel_sizes=KERNEL_SIZE,
                                            rows=utils.approx_index,cols=utils.approx_index)
                COMPLAYER = utils.CompensationLayer(rows=utils.approx_index,cols=utils.approx_index)

                # FIT the MODEL
                HIST = MODEL.fit(x=X_train,y=y_train,batch_size=128,
                                    epochs=n_epochs,verbose=0)             # train model
               
                # RUN & TIME PREDICTIONS
                t0 = time.perf_counter()
                PREDICTIONS = MODEL.predict(x=X_test,batch_size=128,verbose=0)
                t1 = time.perf_counter() 
                predictTime = t1 - t0

                # RUN & TIME COMPENSATION
                t0 = time.perf_counter()
                Xout = COMPLAYER.call(X_test)
                t1 = time.perf_counter()
                compensationTime = t1 - t0

                # ADD DATA TO ARRAYS            
                like_model_data = np.append(like_model_data,predictTime)        # add prediction time to array
                like_model_data = np.append(like_model_data,compensationTime)   # add compensation time to array

            # Compute Averages of N_iters Models
            like_model_data = like_model_data.reshape(N_iters,-1)
            stats = np.mean(like_model_data,axis=0)                 # avg over N_iters
            print("\t\t\t",stats)
                
            row = pd.DataFrame(data=[[model_name,stats[0],stats[1]]],
                                columns=utils.dataframe_cols)                        # make frame
            row.to_csv(outpath,columns=utils.dataframe_cols,header=False,mode='a')  # append to file


    

