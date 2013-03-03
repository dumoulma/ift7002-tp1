'''
Created on Mar 2, 2013

@author: MathieuXPS
'''
import numpy as np
from math import log, sqrt
from functools import reduce

def factorial(n): 
    if n < 2: return 1
    return reduce(lambda x, y: x * y, range(2, int(n) + 1))

def binomial(n, k, r):
    prob = 0
    k = int(k)
    for i in range(0, k + 1):
        prob += factorial(n) / (factorial(k) * factorial(n - k)) * r ** i * (1 - r) ** (n - i)
    return prob

def invbin(n, nRz, delta):
    i = 0
    r_range = np.arange(0,nRz,nRz/100)
    prob = binomial(n, nRz, r_range[i])
    while prob >= delta and i < r_range.size:
        i += 1        
        prob = binomial(n, nRz, r_range[i])
        
    return r_range[i]

def confint_binomial(n, k, delta):
    lower = invbin(n, k, 1 - delta)
    upper = invbin(n, k, delta)
    return lower, upper

z_table = {0.5:0.67, 0.68:1, 0.80:1.28, 0.90:1.64, 0.95:1.96, 0.98:2.33, 0.99:2.58}
def confint_normal(n, Rt, delta):
    z = z_table[1 - delta]
    return z * sqrt((1 / n) * (Rt) * (1 - Rt))

def confint_hoeffding(n, delta):
    delta = 0.1
    return sqrt((1 / (2 * n)) * log(2 / delta))
    
n = 500
k = 25
delta = 0.1
u,l = confint_binomial(n,k,delta)
print("Confint=[%0.5f,%0.5f]"%(l,u))