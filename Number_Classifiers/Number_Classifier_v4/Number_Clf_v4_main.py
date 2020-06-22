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

import Number_Clf_v4_utilities as utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    home_path = 'C:/Users/Landon/Documents/GitHub/Convolutional-Neural-Networks/Number_Classifier_v4'

    MODEL_LAYER_SIZES = utils.N_layer_models    # model sizes dictionary
    X_train,X_test,y_train,y_test = \
        utils.train_test_data(test=0.375,seed=0)
    
    test_type = 'gaussian_noise_v1'        # modifcations made
    N_iterations = 100                      # time to repeat each model

    for N_LAYERS in MODEL_LAYER_SIZES.keys():         # for each MLP size
        print("Testing:",str(N_LAYERS))                 # meesage to user

        out_path = home_path + '/'+ str(N_LAYERS)
        try:
            os.mkdir(out_path)
        except:
            pass

        for N_NUERONS in MODEL_LAYER_SIZES[N_LAYERS]: # for each layer size
            print("\tLayer Sizes:",N_NUERONS)
            print("\t\t\tProgam time:",time.process_time())

            output_matrix = np.array([])                # output for 100 samples 
            for I in range (0,N_iterations,1):

                # Create & Train Model
                t0 = time.process_time()
                model = utils.Create_MLP_Model(name=test_type+'_'+str(I),
                                               layers=N_NUERONS,
                                               X=X_train,y=y_train)
                tf = time.process_time()
                dt = np.round(tf-t0,decimals=8)         # time to create & train model
                print("\t\tTime to train",model.name,":",dt)
                
                # Evaluate Model (scores & confmat)
                model = utils.Eval_MLP_Model(model=model,
                                                X=X_test,y=y_test)
                model = utils.confusion_matrix(model,y_test,
                                               model.predictions,show=False)
                
                # Add to output array
                row = np.array([dt,model.loss_,model.n_iter_])
                row = np.append(row,model.precision_scores)
                row = np.append(row,model.recall_scores)
                row = np.append(row,model.confusion_matrix)

                output_matrix = np.append(output_matrix,row)    
                del(model)              # delete model to save RAM

            # reshape output matrix
            output_matrix = output_matrix.reshape(N_iterations,-1)
            output_name = test_type+'_'+str(N_NUERONS[0])
            Frame = utils.Create_DataFrame(output_matrix)
            Frame.to_csv(out_path+'/'+output_name+'.csv')

    print("Total program Time:",time.process_time())
                



        

