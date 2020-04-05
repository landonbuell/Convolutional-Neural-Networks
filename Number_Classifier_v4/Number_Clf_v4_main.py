"""
Landon Buell
Qiaoyan Yu
Number Classifier v4 - main
3 April 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import time

import Number_Clf_v4_utilities as utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    MODEL_SIZES = utils.N_layer_models         # model sizes dictionary
    X_train,X_test,y_train,y_test = \
        utils.train_test_data(test=0.375,seed=0)
    
    test_type = 'control_v1'        # modifcations made
    N_iterations = 100              # time to repeat each model

    for model_type in MODEL_SIZES.keys():               # for each MLP size
        print("Testing:",str(model_type))               # meesage to user

        for layer_sizes in MODEL_SIZES[model_type]:     # for each layer size
            print("\tLayer Sizes:",layer_sizes)

            for I in range (0,N_iterations,1):
                
                t0 = time.process_time()

                model = utils.Create_MLP_Model(name=)

                tf = time.process_time()



        

