"""
Landon Buell
Float to Binary
"""

import numpy as np
import bitstring

# Create W,x,b

x0 = np.random.randn(10,1)*10
W0 = np.random.randn(8,10)*10
b0 = np.random.randn(8,1)*10

x1 = W0 @ x0 + b0

# Examine activations
a1 = W0 @ x0
print(a1)

def Swap_MSB_LSB (act):
    """ Sweap MSB & LSB in exponente of IEEE 754 FP-64 """
    act = act.ravel()           # flatten arr
    for I in range(len(act)):               # each entry in arr
        bin_str = bitstring.BitArray(float=act[I],length=64).bin # binary str
        bin_list = list(bin_str)                             # convert to list
        bin_list[1],bin_list[11] = bin_list[11],bin_list[1] # swap bits
        new_str = ''.join(bin_list)                         # back to str
        new_float = bitstring.BitString(bin=new_str).float  # convert to float 64
        act[I] = new_float                                  # overwrite index
    return act.reshape(-1,1)                                # return array

c1 = Swap_MSB_LSB(a1)

print(c1)