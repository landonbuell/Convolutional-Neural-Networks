"""
Landon Buell
Qiaoyan Yu
Number Classifier v4 - main
3 April 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import time

import NUM_CLF_v5_Utilities as utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    home_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v5/Number_Classifier_v5'
    data_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v5/Raw_Data'

    MODEL_LAYER_SIZES = utils.N_layer_models    # model sizes dictionary
    frame_cols = utils.dataframe_columns
    X_train,X_test,y_train,y_test = \
        utils.train_test_data(test=0.375,seed=0)
    
    test_type = 'Mute_MSB_v3'           # modifcations made
    N_iterations = 50                   # time to repeat each model
    output_frame = pd.DataFrame(data=None,columns=frame_cols)
    output_frame.to_csv(path_or_buf=data_path+'/'+test_type+'.csv',
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
                model = utils.Create_MLP_Model(name=test_type+'_'+str(I),
                                    layers=NEURON_DENSITY,X=X_train,y=y_train)              
                model = utils.Eval_MLP_Model(model=model,X=X_test,y=y_test)

                # Create row to to add to matrix
                row = np.array([model.loss_,model.n_iter_,
                                model.avg_prec,model.avg_recall])
                output_matrix = np.append(output_matrix,row)                  
                del(model)              # delete model to save RAM

            # reshape output matrix,concatenate dataframes
            output_matrix = output_matrix.reshape(N_iterations,-1)
            frame = utils.Create_DataFrame(output_matrix,model_name,frame_cols)
            output_frame = pd.concat([output_frame,frame])
            output_frame.to_csv(path_or_buf=data_path+'/'+test_type+'.csv',
                        header=True,mode='w')       # overwrite the output frame

    print("Total program Time:",time.process_time())
                



        


