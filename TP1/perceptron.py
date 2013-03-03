'''
Created on Mar 2, 2013

@author: MathieuXPS
'''
import file_loader
from helpers import sng, compute_gram
import numpy as np
    
class Perceptron(object):
    '''
        Implementatino of the dual Perceptron with rbd kernel linear Classification algorithm
        
        Use:
    '''

    def __init__(self, sigma=1, kernel='rbf'):
        self.sigma = sigma        
    
    def fit(self, X, y):
        m = X.shape[0]
        self._y = y
        nCols = X.shape[1] + 1
        self._u = np.ones(shape=(m, nCols), dtype=np.float64)
        for j in range(1, nCols):
            self._u[:, j] = X[:, j - 1]

        self.G = compute_gram(self._u, y)        
        self.alpha = np.zeros(shape=(m,), dtype=np.int32)
        wasUpdated = True
        while wasUpdated:
            wasUpdated = False
            for i in range(m):
                isCorrect = sum([self.G[i, j] * self.alpha[j] for j in range(m)]) <= 0
                if not isCorrect:
                    self.alpha[i] += 1
                    wasUpdated = True
    
    def predict(self, X):
        m = X.shape[0]
        nCols = X.shape[1] + 1
        u = np.ones(shape=(m, nCols), dtype=np.float64)
        for j in range(1, nCols):
            u[:, j] = X[:, j - 1]
        predictions = np.zeros(m,)
        for i in range(m):
            predictions[i] = sum([self.alpha[j] * self._y * np.dot(self._u[j], u[i]) for j in range(m)])
            pass
        return sng(predictions)
    
X_s, y_s, X_t, y_t = file_loader.fetch_mnist()
clf = Perceptron()
clf.fit(X_s, y_s)
