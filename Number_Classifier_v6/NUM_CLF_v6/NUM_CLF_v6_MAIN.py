"""
Landon Buell
Number Classifier v6
Main Script
7 June 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.keras as keras

import NUM_CLF_v6_Utilities as utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    home_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v6/NUM_CLF_v6'
    data_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v6/Raw_Data'
    test_name = 'Baseline'
    MODEL_LAYER_SIZES = utils.N_layer_models    # model sizes dictionary
    frame_cols = utils.dataframe_columns

    X_train,X_test,y_train,y_test = utils.Load_MNIST()
    X_train *= 255
    X_test *= 255
    y_train = keras.utils.to_categorical(y_train,10)
    y_test = keras.utils.to_categorical(y_test,10)

    N_iterations = 50       # repeat each model
    output_frame = pd.DataFrame(data=None,columns=frame_cols)
    output_frame.to_csv(path_or_buf=data_path+'/'+test_name+'.csv',
                        mode='w')       # write the output frame

    for N_LAYERS in MODEL_LAYER_SIZES.keys():         # for each MLP size
        print("Testing:",str(N_LAYERS))                 # meesage to user

        for NEURON_DENSITY in MODEL_LAYER_SIZES[N_LAYERS]: # for each layer size
            print("\tLayer Sizes:",NEURON_DENSITY)
            print("\t\t\tProgam time:",time.process_time())

            model_name = N_LAYERS+'_'+str(NEURON_DENSITY[0])

            output_matrix = np.array([])                # output for N samples 
            for I in range (0,N_iterations,1):
                print("\t\tIteration:",I)
                # Create & Train Model
                MODEL = utils.Keras_Model(name=test_name+'_'+str(I),
                                          layers=NEURON_DENSITY)    
                TRAINING_HISTORY = MODEL.fit(X_train,y_train,batch_size=100,epochs=200,verbose=0)
                #utils.Plot_History(TRAINING_HISTORY,MODEL,show=True)
                MODEL = utils.Eval_Model(MODEL,X_test,y_test)

                # Create row to to add to matrix
                row = np.array([MODEL.loss,
                                MODEL.avg_prec,MODEL.avg_recall])
                output_matrix = np.append(output_matrix,row)                  
                del(MODEL)              # delete model to save RAM

            # reshape output matrix,concatenate dataframes
            output_matrix = output_matrix.reshape(N_iterations,-1)
            frame = utils.Create_DataFrame(output_matrix,model_name,frame_cols)
            output_frame = pd.concat([output_frame,frame])
            output_frame.to_csv(path_or_buf=data_path+'/'+test_name+'.csv',
                        header=True,mode='w')       # overwrite the output frame

    print("Total program Time:",time.process_time())