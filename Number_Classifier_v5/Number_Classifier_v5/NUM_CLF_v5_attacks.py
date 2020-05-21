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
 
        #### ATTACK FUNCTIONS ####

def Swap_MSB_LSB (act):
    """ Sweap MSB & LSB in exponente of IEEE 754 FP-64 """
    act_shape = act.shape                   # original shape
    act = act.ravel().astype(np.float64)    # flatten,make FP-64 if not
    for I in range(len(act)):               # each entry in arr
        bin_str = bitstring.BitArray(float=act[I],length=64).bin # binary str
        bin_list = list(bin_str)                            # convert to list
        bin_list[12],bin_list[63] = \
            bin_list[63],bin_list[12]                       # swap bits
        new_str = ''.join(bin_list)                         # back to str
        new_float = bitstring.BitString(bin=new_str).float  # convert to float 64
        act[I] = new_float                                  # overwrite index
    act = act.reshape(act_shape)
    return act

def Mute_MSB (act):
    """ Mute Most-Signfigicant bit in exponet of FP-64 """
    act_shape = act.shape                   # original shape
    mants,exps = np.frexp(act)              # mantissa,exponent
    exps = np.array([0 if (e > 0) else e for e in exps.ravel()])
    act = np.ldexp(mants,exps.reshape(act_shape))   # reconstruct FP-64
    return act                                  # return new activations

def round (act,decimals=0):
    """ Round elements of matrix product to specified decimal accuracy """
    return np.round(act,decimals=decimals)  # return rounded vector

def gaussian_noise(activations):
    """ Add gaussian noise to activations """
    act_shape = activations.shape       # shape of array
    noise = np.random.normal(loc=0,scale=2,size=act_shape)
    return activations + noise          # add noise to array

        #### TRIGGER FUNCTIONS ####

def get_trigger(trigger_type=None):
    """ Return boolean trigger condition """
    if trigger_type == 'binary':
        return np.random.choice([True,False],p=[0.5,0.5])
    elif trigger_type == 'always_on':
        return True
    elif trigger_type == None:
        return False
    else:
        return False


        #### MAIN ATTACK FUNCTION ####
   
def ATTACK (activations,attack_type=None,trigger_type=None):
    """Simulate Attack Function """
    
    # Get trigger condition
    trigger = get_trigger(trigger_type=trigger_type)
    
    # if true, apply attack
    if trigger == True:

        if attack_type == 'round':
            return round(activations,decimals=0)
            
        elif attack_type == 'noise':
            return gaussian_noise(activations)
            
        elif attack_type == 'mute_MSB':
            return Mute_MSB(activations)
            
        else:
            return activations
        
    else:
        return activations
