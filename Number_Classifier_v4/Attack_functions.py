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


def round (act,dcmls=0):
    """ Round elements of matrix product to specified decimal accuracy """
    return np.round(act,decimals=dcmls)     # return rounded vector



def ATTACK (a,b,attack_type=None):
    """
    Simulate and attack fucntion
    """
    if attack_type == None:         
        # No attack, just matrix multiply
        return a @ b

    if attack_type == 'round_activatons':     
        # Rounding attack, round to 0 decimals
        activation = a @ b
        return round(activation,)

    if a