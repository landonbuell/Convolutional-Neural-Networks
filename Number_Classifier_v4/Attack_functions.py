"""
Landon Buell
Qiaoyan Yu
Number Classifier v4 - main
3 April 2020
"""

        #### IMPORTS ####

import numpy as np
import os
import sys

"""
Number Classifier Attack Functions
    All functions are g(W,x,t,*params) 
        - W is weighting matrix
        - x is a coluimn vecotr (layer of network)
        - t is a boolean activation condition
        - *params are unique to particular function
"""

def ATTACK (func):
    """

    """
    return func(a,b,trigger)


def round (W,x,trigger):
    """
    Round elements of matrix product to specified decimal accuracy
    --------------------------------

    --------------------------------
    Return attack function modifcation to matrix product
    """
    if trigger == False:
        return W @ x



