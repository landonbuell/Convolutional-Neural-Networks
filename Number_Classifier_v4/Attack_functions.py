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

        #### ATTACK FUNCTIONS ####

def round (act,decimals=0):
    """ Round elements of matrix product to specified decimal accuracy """
    return np.round(act,decimals=decimals)  # return rounded vector

def bit_swap(act):
    """ Swap bit in binary equivalent of floating point number """
    decimal_places = 8   
    return activations 

        #### TRIGGER FUNCTIONS ####

def get_trigger (trigger_type):
    """ Compute & return boolean trigger value """
    if trigger_type == 'binary':
        return np.random.choice([True,False],size=1,p=[0.5,0.5])

    
def ATTACK (activations,attack_type=None,trigger_type='binary'):
    """
    Simulate and attack fucntion
    """
    
    # get boolean trigger value
    trigger_condition = get_trigger(trigger_type=trigger_type)

    if trigger_condition == True:       # if trigger active

        if attack_type == 'round_activatons':     
            # Rounding attack, round to 0 decimals
            return round(activations,decimals=0)

        if attack_type == 'bit_swap':
            # swap bit in binary equivalent
            return bit_swap(activations)

        else:
            # no attck type, no change:
            return activations

    else:
        return activations