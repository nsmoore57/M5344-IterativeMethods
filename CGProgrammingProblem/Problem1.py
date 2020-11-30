import time
import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
import scipy.sparse as sp
from scipy.linalg import norm
from scipy.sparse.linalg import splu
import scipy.io
from PCG import PCG
from BasicPreconditioner import *
from string import Template
from KrylovUtils import *
import gc

# Set up random state
rs = RandomState(MT19937(SeedSequence(123456789)))

for i in range(9,21):
    # read in a matrix
    A = scipy.io.mmread(f'../TestMatrices/DH-Matrix-{i}.mtx')
    A = A.tocsr()

    n, nc = A.shape
    print(f'Size: {n}\\times{nc}')

    xTrue = rs.rand(n)
    b = A*xTrue

    tols = [1.0e-6, 1.0e-8]
    drops = []
    # if n < 10**5:
    #     drops.extend([10**(-j) for j in range(2)])
    drops.extend([10**(-j) for j in range(2,5)])

    # Do a direct solve for comparison - same for all err tolerance and drop tolerance
    startT = time.time()
    LU = splu(A.tocsc())
    x_Direct = LU.solve(b)
    Direct_time = time.time() - startT

    err_Direct = norm(x_Direct - xTrue)/norm(xTrue)
    resid_Direct = norm(b-mvmult(A,x_Direct))/norm(b)

    del LU

    for t in tols:
        print(f'\nStopping Tolerance: {t}\n')

        for d in drops:
            # Build the preconditioner
            startT = time.time()
            ILU = ILURightPreconditioner(A, drop_tol=d, fill_factor=30)
            ILU_time = time.time() - startT

            # Run CG
            startT = time.time()
            (conv, iters, x, _, _) = PCG(A, b, maxiter=10000, tau=t, precond=ILU)
            CG_time = time.time() - startT

            # get the residual and error
            resid_CG = norm(b - mvmult(A,x))/norm(b)
            err_CG = norm(x - xTrue)/norm(xTrue)

            print(f'{d:g} & {iters} & {resid_CG:1.2e} & {err_CG:1.2e} & {ILU_time:10.3g} & {CG_time:10.3g} & {ILU_time + CG_time:10.3g} & {resid_Direct:1.2e} & {err_Direct:1.2e} & {Direct_time:10.3g}\\\\')
            print('\\hline')
            del ILU
            del x
            gc.collect()
