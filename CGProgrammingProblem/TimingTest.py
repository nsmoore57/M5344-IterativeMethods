import time
import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
import scipy.sparse as sp
from scipy.linalg import norm
import scipy.sparse.linalg as splu
import scipy.io
import matplotlib.pyplot as plot
from PCG import PCG
from BasicPreconditioner import *
from string import Template
from KrylovUtils import *
import gc

# Set up random state
rs = RandomState(MT19937(SeedSequence(123456789)))

# testMatrices = [6, 10, 12, 14, 15]
testMatrices = [6, 10, 12, 14, 16, 18, 20]

# Run unpreconditioned CG on the matrices, collecting the results for relative Residual and relative Error
for (i,m) in enumerate(testMatrices):
    # read in a matrix
    A = scipy.io.mmread(f'../TestMatrices/DH-Matrix-{m}.mtx')
    A = A.tocsr()
    print(f'Matrix {m}')

    # Get the largest and smallest e-val
    RatioTimer = MyTimer('Ratio')
    evl, _ = splu.eigs(A, which='LM')
    evs, _ = splu.eigs(A, sigma=1e-8)
    evl = abs(evl)
    evs = abs(evs)
    condNumRatio = evl.max()/evs.min()
    RatioTimer.stop()
    print(f'Ratio:  {condNumRatio} in {RatioTimer.walltime()}')

    del A
    gc.collect()
