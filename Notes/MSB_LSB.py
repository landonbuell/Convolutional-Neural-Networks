"""
Landon Buell
Float to Binary
"""

import numpy as np
import bitstring
import time

# create activations

a1 = np.random.randn(100,10)*10

def Mute_Bits (act,bits):
    """ Mute bits to 0 indicated by elements in list """
    act_shape = act.shape                   # original shape
    act = act.ravel().astype(np.float64)    # flatten,make FP-64 if not
    for I in range(len(act)):               # each entry in arr
        bin_str = bitstring.BitArray(float=act[I],length=64).bin # binary str
        bin_list = list(bin_str)                            # convert to list
        for bit in bits:                                    # for each bit
            bin_list[bit] = '0'                             # drop to 0
        new_str = ''.join(bin_list)                         # back to str
        new_float = bitstring.BitString(bin=new_str).float  # convert to float 64
        act[I] = np.float64(new_float)                      # overwrite index
    act = act.reshape(act_shape)            # reshape
    return act                              # return 

t1 = time.process_time_ns()

c1 = Mute_Bits(a1,bits=[1,12])

t2 = time.process_time_ns()

print(t2-t1)

#print(c1)