"""
Landon Buell
Qioyan Yu
Error-Comp-v1
6 July 2020
"""
        
            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras

import ERR_COMP_v1_Utilities as utils      

            #### MAIN EXECUABLE ####

if __name__ == '__main__':

       
        # PRE-PROCESSING 
        init_path = os.getcwd()
        X_train,y_train,X_test,y_test = utils.Load_CIFAR10()
        LAYER_MODELS = utils.N_layer_models
        y_train = keras.utils.to_categorical(y_train,10) 
        y_test = keras.utils.to_categorical(y_test,10) 

        # DIRECTORY
        output_frame = pd.DataFrame(columns=utils.dataframe_cols)   # create dataframe
        outpath = os.path.join(utils.output_path,utils.outfile_name)
        if os.path.isfile(outpath):     # file exists
            pass                        # do nothing
        else:                           # does not exist
            output_frame.to_csv(outpath,
                        columns=utils.dataframe_cols,mode='w')      # make new file

        N_iters = 4             # Time to repeat each model
        n_epochs = 10           # epochs over data set

        utils.Plot_Sample(X_test[1],' ')

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
                    t_0 = time.perf_counter()
                    HIST = MODEL.fit(x=X_train,y=y_train,batch_size=128,
                                     epochs=n_epochs,verbose=0)             # train model
                    t_f = time.perf_counter()
                    traintime = np.round(t_f-t_0,decimals=4)
                    METRICS = MODEL.evaluate(x=X_test,y=y_test,batch_size=128,
                                            verbose=0)
                    like_model_data = np.append(like_model_data,METRICS)    # add metrics to array
                    like_model_data = np.append(like_model_data,traintime)  # add fitting time to array

                # Compute Averages of N_iters Models
                like_model_data = like_model_data.reshape(N_iters,-1)
                stats = np.mean(like_model_data,axis=0)                 # avg over N_iters
                print("\t\t\t",stats)
                
                row = pd.DataFrame(data=[[model_name,stats[0],stats[1],stats[2],stats[3]]],
                                   columns=utils.dataframe_cols)                        # make frame
                row.to_csv(outpath,columns=utils.dataframe_cols,header=False,mode='a')  # append to file



