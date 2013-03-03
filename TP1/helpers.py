'''
Created on Mar 2, 2013

@author: MathieuXPS
'''
import numpy as np
from math import exp
import numpy.linalg as LA

def compute_gram(u, y):
    m = u.shape[0]
    G = np.zeros(shape=(m, m), dtype=np.float64)    
    for i, j in [(i, j) for i in range(m) for j in range(m)]:
        G[i, j] = y[i] * y[j] * np.dot(u[i, :], u[j, :])
    return G

def rbg_kernel(x1, x2, sigma=1):
    return exp(-1 * (LA.norm(x1 - x2) ** 2) / (2 * sigma ** 2))

def sng(aVector):
    pred = np.array(aVector, dtype=np.int)
    pred[aVector <= 0] = -1
    pred[aVector > 0] = 1
    return pred
