# -*- coding: utf-8 -*-

import numpy as np

num = int(input("Enter num:" ))

def warmup(x):
    return np.identity(x)
    

print(warmup(num))    