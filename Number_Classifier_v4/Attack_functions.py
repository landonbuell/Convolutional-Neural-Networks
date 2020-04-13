"""
Landon Buell
Qiaoyan Yu
Number Classifier v4 - main
3 April 2020
"""

        #### IMPORTS ####

import numpy as np
import struct

"""
Number Classifier Attack Functions
    All functions are g(W,x,t,*params) 
        - W is weighting matrix
        - x is a coluimn vecotr (layer of network)
        - t is a boolean activation condition
        - *params are unique to particular function
"""
        #### MISC FUNCTIONS ####

""" These three functions were modifed from an online source. See:
https://www.technical-recipes.com/2012/converting-between-binary-and-decimal-representations-of-ieee-754-floating-point-numbers-in-c/ """

getBin = lambda x: x > 0 and str(bin(x))[2:] or "-" + str(bin(x))[3:]

def FloatToBinary64(float_val):
    """ Convert floating point number to 64-bit Binary String"""
    val = struct.unpack('Q', struct.pack('d', float_val))[0]
    return getBin(val)
       
def Binary64ToFloat(binary_val):
    """ Convert 64-bit Binary String to floating point number"""
    hx = hex(int(binary_val, 2))   
    return struct.unpack("d", struct.pack("q", int(hx, 16)))[0]


        #### ATTACK FUNCTIONS ####

def round (act,decimals=0):
    """ Round elements of matrix product to specified decimal accuracy """
    return np.round(act,decimals=decimals)  # return rounded vector

def mute_bits(act):
    """ Swap bit in binary equivalent of floating point number """
    orignal_shape = act.shape     # original shape of arr
    act = act.ravel()               # flatten
    for neuron in act:
        binary_string = FloatToBinary64(neuron)         # convert to 64-bit binary
        binary_list = list(binary_string)               # convert to list
        pts = np.random.randint(low=0,high=63,size=4)   # generate 4 rand idxs
        for pt in pts:                                  # for the 4 random pts
            binary_list[pt] = '0'                       # mute the bit to '0'
        binary_string =  ''.join(binary_list)           # rejoin into single string
        neuron = Binary64ToFloat(binary_string)         # replace neuron value
    act = act.rehape(orignal_shape)         # rehape to original
    return act                      # return the activations

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
    
    # get boolean trigger value
    trigger_condition = get_trigger(trigger_type=trigger_type)
    
    if trigger_condition == True:       # if trigger active

        if attack_type == 'round_activatons':     
            # Rounding attack, round to 0 decimals
            return round(activations,decimals=0)

        if attack_type == 'mute_bits':
            # swap bit in binary equivalent
            return bit_swap(activations)

        else:
            # no attck type, no change:
            return activations

    else:
        return activations