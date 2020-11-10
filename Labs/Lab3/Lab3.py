import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import random


class SmoAlgorithm:

    def __init__(self, X, Y, kernel_function, C, tol=0.001):
        self.X = X
        self.Y = Y
        self.M, self.N = np.shape(self.X)

        self.kernel_function = kernel_function
        self.C = C
        self.kernel_matrix = getKernelMatrix(self.kernel_function, X, X)

        self.alphas = np.zeros(self.M)
        self.B = 0
        self.W = np.zeros(self.N)

        self.errors = np.zeros(self.M)
        self.epsilon = 1e-3
        self.tol = tol

        self.MAX_ITERATIONS = 3000

    def predict(self, x):
        res =


def getKernelMatrix(kernel_function, A, B):
    n, *_ = A.shape
    m, *_ = B.shape
    f = lambda i, j: kernel_function(A[i], B[j])
    return np.fromfunction(np.vectorize(f), (n, m), dtype=int)
