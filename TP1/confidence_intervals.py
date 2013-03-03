'''
Created on Mar 2, 2013

@author: MathieuXPS
'''
import numpy as np
from math import log, sqrt
from functools import reduce
from rpy import *

def __factorial(n): 
    if n < 2: return 1
    return reduce(lambda x, y: x * y, range(2, int(n) + 1))

def __binomial(n, k, r):
    prob = 0
    k = int(k)
    for i in range(0, k + 1):
        prob += __factorial(n) / (__factorial(k) * __factorial(n - k)) * r ** i * (1 - r) ** (n - i)
    return prob

def __invbin(n, nRz, delta):
    r_max = 0
    r_range = np.arange(0, nRz, nRz / 100)
    i = 0
    prob = rpy.pbinom(n, nRz, r) 
    while prob >= delta and i < r_range.size:
        r_val = r_range[i]
        r_max = r_val
        i += 1
        
    return r_max

def confint_binomial(n, k, delta):
    lower = __invbin(n, k, 1 - delta)
    upper = __invbin(n, k, delta)
    return lower, upper

z_table = {0.5:0.67, 0.68:1, 0.80:1.28, 0.90:1.64, 0.95:1.96, 0.98:2.33, 0.99:2.58}
def confint_normal(n, Rt, delta):
    z = z_table[1 - delta]
    return z * sqrt((1 / n) * (Rt) * (1 - Rt))

def confint_hoeffding(n, delta):
    delta = 0.1
    return sqrt((1 / (2 * n)) * log(2 / delta))
    
