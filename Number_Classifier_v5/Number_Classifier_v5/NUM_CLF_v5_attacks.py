"""
Landon Buell
Qiaoyan Yu
NUM ClF
24 April 2020
"""

        #### IMPORTS ####

import numpy as np


"""
Number Classifier Attack Functions
    activations (arr) : (1 x M) column vector that results from matrix vector multiplication
    attack_type (str) : character string indictaing the type of attack to be performed (None by default)
    trigger_type (str) : trigger condition to used for attack frequnecy ('binary' by default)
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
    return struct.unpack("d", struct.pack("f", int(hx, 16)))[0]


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

        if attack_type == 'mute_bits':
            # swap bit in binary equivalent
            return mute_bits(activations)

        if attack_type == 'gaussian':
            return gaussian_noise(activations)

        else:
            # no attck type, no change:
            return activations

    else:
        return activations
