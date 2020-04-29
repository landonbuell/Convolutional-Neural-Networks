"""
Landon Buell
Qiaoyan Yu
NUM ClF
24 April 2020
"""

        #### IMPORTS ####

import numpy as np
import bitstring

"""
Number Classifier Attack Functions
    activations (arr) : (1 x M) column vector that results from matrix vector multiplication
    attack_type (str) : character string indictaing the type of attack to be performed (None by default)
    trigger_type (str) : trigger condition to used for attack frequnecy ('binary' by default)
"""
        #### MISC FUNCTIONS ####



        #### ATTACK FUNCTIONS ####

def Swap_MSB_LSB (act):
    """ Sweap MSB & LSB in exponente of IEEE 754 FP-64 """
    act_shape = act.shape           # original shape
    act = act.ravel()               # flatten arr
    for I in range(len(act)):       # each entry in arr
        bin_str = bitstring.BitArray(float=act[I],length=64).bin # binary str
        bin_list = list(bin_str)                             # convert to list
        bin_list[1],bin_list[11] = bin_list[11],bin_list[1] # swap bits
        new_str = ''.join(bin_list)                         # back to str
        new_float = bitstring.BitString(bin=new_str).float  # convert to float 64
        act[I] = new_float                                  # overwrite index
    return act.reshape(act_shape)                           # return array

def round (act,decimals=0):
    """ Round elements of matrix product to specified decimal accuracy """
    return np.round(act,decimals=decimals)  # return rounded vector

def gaussian_noise(activations):
    """ Add gaussian noise to activations """
    act_shape = activations.shape       # shape of array
    noise = np.random.normal(loc=0,scale=1,size=act_shape)
    return activations + noise          # add noise to array

        #### TRIGGER FUNCTIONS ####

def get_trigger (trigger_type):
    """ Compute & return boolean trigger value """
    if trigger_type == 'binary':
        return np.random.choice([True,False],size=1,p=[0.5,0.5])
    if trigger_type == 'always_on':
        return True
    else: 
        return False


        #### MAIN ATTACK FUNCTION ####
   
def ATTACK (activations,attack_type=None,trigger_type='binary'):
    """
    Simulate and attack fucntion
    """

    if attack_type == None:     # no attack
        return activations      # untouched activations
    
    # get boolean trigger value
    trigger_condition = get_trigger(trigger_type=trigger_type)
    
    if trigger_condition == True:       # if trigger active

        if attack_type == 'round_activatons':     
            # Rounding attack, round to 0 decimals
            return round(activations,decimals=0)

        if attack_type == 'swap_bits':
            # swap bit in binary equivalent
            return Swap_MSB_LSB(activations)

        if attack_type == 'gaussian':
            return gaussian_noise(activations)

        else:
            # no attck type, no change:
            return activations

    else:
        return activations
