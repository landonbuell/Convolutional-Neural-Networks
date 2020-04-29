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
a1 = a1.ravel()
print(a1)

# Print 0th idx
print(type(a1[0]))
print(a1[0])

# convert entry to single-precision float
f1 = bitstring.BitArray(float=a1[0],length=64)
f1_bin = f1.bin
print(f1_bin)

# convert str to lists?
print(type(f1_bin))
f1_list = list(f1_bin)
print(f1_list)

f1_list[11],f1_list[1] = f1_list[1],f1_list[11]

print(f1_list)

f1_new = ''.join(f1_list)
print(f1_new)

print(f1_bin)

f1 = bitstring.BitString(bin=f1_new)

print(f1.float)