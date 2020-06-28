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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras

import ERR_COMP_Utilities as utils

            
            #### MAIN EXECUABLE ####

if __name__ == '__main__':

       
        # PRE-PROCESSING 
        init_path = os.getcwd()
        X_train,y_train,X_test,y_test = utils.Load_CIFAR10()
        LAYER_MODELS = utils.N_layer_models
        y_train = keras.utils.to_categorical(y_train,10) 
        y_test = keras.utils.to_categorical(y_test,10) 
        output_frame = pd.DataFrame(columns=utils.dataframe_cols)

        N_iters = 4             # Time to repeat each model
        n_epochs = 10           # epochs over data set

        #utils.Plot_Sample(X_test[1],' ')

        # APPROXIMATIONS
        print("Approximating Data...\n")
        ApproxLayer = utils.ApproximationLayer(rows=utils.approx_index,
                                               cols=utils.approx_index)
        #X_train = ApproxLayer.call(X_train)
        #X_test = ApproxLayer.call(X_test)
        #utils.Plot_Sample(X_test[1],' ')

        # COMPENSATIONS
        print("Compensating Data...\n")
        CompLayer = utils.CompensationLayer(rows=utils.approx_index,
                                            cols=utils.approx_index)

        #X_train = CompLayer.call(X_train)
        #X_test = CompLayer.call(X_test)
        #utils.Plot_Sample(X_test[1],' ')

        # ITERATE BY LAYER
        for N_LAYERS in LAYER_MODELS.keys():            # Each number of layers
            print('Layers: '+str(N_LAYERS)+'-',time.perf_counter())     # indicate layers

            for KERNEL_SIZE in LAYER_MODELS[N_LAYERS]:    # Each number of nodes
                print('\tDensity: '+str(KERNEL_SIZE[0])+'-',time.perf_counter())  #inicate nodes
                model_name = str(N_LAYERS)+'_'+str(KERNEL_SIZE[0])
                like_model_data = np.array([])          # hold dat from each iter
                
                for i in range(N_iters):            # Each Iteration
                    print('\t\tIteration: '+str(i)+'-',time.perf_counter())

                    MODEL = utils.Network_Model(name=model_name,kernel_sizes=KERNEL_SIZE)
                    HIST = MODEL.fit(x=X_train,y=y_train,batch_size=128,
                                     epochs=n_epochs,verbose=2)             # train model
                    METRICS = MODEL.evaluate(x=X_test,y=y_test,batch_size=128,
                                            verbose=2,)
                    like_model_data = np.append(like_model_data,METRICS)    # add metrics to array

                # Compute Averages of N_iters Models
                like_model_data = like_model_data.reshape(N_iters,-1)
                stats = np.mean(like_model_data,axis=0)                 # avg over N_iters
                print("\t\t\t",stats)
                
                row = pd.DataFrame(data=[[model_name,stats[0],stats[1],stats[2]]],
                                   columns=utils.dataframe_cols)
                output_frame = output_frame.append(row,ignore_index=True)
            
            # Update output file
            outpath = utils.output_path+'/'+utils.outfile_name
            output_frame.to_csv(outpath,columns=utils.dataframe_cols,mode='w')

