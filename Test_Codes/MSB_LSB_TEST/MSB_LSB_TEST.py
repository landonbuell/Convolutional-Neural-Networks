"""
Landon Buell
Mute MSB test
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':

    x0 = np.random.randn(10,100)*10
    shape = x0.shape

    t_0 = time.process_time_ns()
    mants,exps = np.frexp(x0)
    exps = np.array([0 if (e > 0) else e for e in exps.ravel()])
    x1 = np.ldexp(mants,exps.reshape(shape))
    t_f = time.process_time_ns()

    print(t_f - t_0)

