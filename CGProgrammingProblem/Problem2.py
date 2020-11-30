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

testMatrices = [6, 10, 12, 14, 16, 18, 20]

relResid = [[] for i in testMatrices]
relError = [[] for i in testMatrices]
condNums = [0.0 for i in testMatrices]

# Run unpreconditioned CG on the matrices, collecting the results for relative Residual and relative Error
for (i,m) in enumerate(testMatrices):
    # read in a matrix
    A = scipy.io.mmread(f'../TestMatrices/DH-Matrix-{m}.mtx')
    A = A.tocsr()

    # Print the shape
    n, nc = A.shape
    print(f'Size: {n}\\times{nc}')

    # Get the condition number
    evl, _ = splu.eigs(A, which='LM')
    evs, _ = splu.eigs(A, sigma=1e-8)
    evl = abs(evl)
    evs = abs(evs)
    condNums[i] = evl.max()/evs.min()

    xTrue = rs.rand(n)
    b = A*xTrue

    tol = 1e-14
    maxiter = 10**4
    (conv, iters, _, relResid[i], relError[i]) = PCG(A, b, maxiter=maxiter, tau=tol, xTrue = xTrue)

    del A
    del b
    gc.collect()

# Plotting
###########

# Set up the colors:
# colormap = plot.cm.get_cmap('hsv', len(testMatrices) + 1)
colormap = plot.get_cmap("tab10")

# Now plot all bounds and actual errors in the same semilog plot
for i in range(len(testMatrices)):
    # Set up the n values for the theoretical bound:
    ns = range(len(relError[i]))
    # Calculate the theoretical bound based on the condition number of the matrix
    theoBound = ((np.sqrt(condNums[i]) - 1)/(np.sqrt(condNums[i]) + 1))**ns

    # Plot the theoretical bound
    plot.semilogy(ns, theoBound, color=colormap(i), linestyle='dashed')

    # Plot the actual relative error
    plot.semilogy(ns, relError[i], color=colormap(i), linestyle='solid', label = f'Matrix {testMatrices[i]}')

    if i == 3:
        plot.legend(loc='center right')
        plot.savefig("Problem2-ErrorPlot-First4.pdf")

plot.legend(loc='center right')
plot.savefig("Problem2-ErrorPlot-all.pdf")
# plot.show()
