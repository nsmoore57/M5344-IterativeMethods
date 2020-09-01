#!/usr/bin/python
"""Module for Cholesky Methods"""
import numpy as np
import scipy.sparse as sp

def CholeskyFactorization(A):
    """
    Runs a Cholesky Factorization on matrix A.

    The matrix A is expected to be:
    - Square
    - Tridiagonal
    - Symmetric
    - Positive Definite
    In the event the matrix fails one of these condition, a RuntimeError is raised.
    """

    # Check if the matrix is in dia_matrix format, and convert otherwise
    if sp.isspmatrix_dia(A):
        T = A
    else:
        T = sp.dia_matrix(A)

    # Check the dimensions
    (m, n) = T.shape
    if m != n:
        raise RuntimeError("Matrix must be square")

    # Check if the matrix is tridiagonal
    if not _isTridiagonal(T):
        raise RuntimeError("Matrix must be tridiagonal")

    # Check for symmetry - we already know it's tridiagonal, only 1 diag above and below to check
    if not all(T.diagonal(-1) == T.diagonal(1)):
        raise RuntimeError("Matrix must be symmetric")

    # Make copies of the necessary diagonals to prevent changing the original matrix
    L_main_diag = T.diagonal(0).copy()
    L_sub_diag = T.diagonal(-1).copy()

    # Build L column by column
    for i in range(n):
        # Only need the subtraction if not the first column
        if i > 0:
            L_main_diag[i] -= L_sub_diag[i-1]*L_sub_diag[i-1]

        # Check if matrix is indefinite here to prevent complex arithmetic cost
        if L_main_diag[i] < 0:
            raise RuntimeError("Matrix must be positive definite")

        # Now that we know it's positive, can take the square root
        L_main_diag[i] = np.sqrt(L_main_diag[i])

        # Subdiagonal is one element shorter than main diagonal so can't assign at i == n-1
        if i < n-1:
            L_sub_diag[i] /= L_main_diag[i]

    L = sp.diags([L_main_diag, L_sub_diag], [0, -1])
    return L

def _isTridiagonal(T):
    """
    Checks to see whether L is tridiagonal.
    L must be in scipy.sparse.dia_matrix form
    """
    # First check if there's 3 diagonals
    if len(T.offsets) != 3:
        return False

    # Check for the main diagonal
    if 0 not in T.offsets:
        return False

    # Check for the super diagonal
    if 1 not in T.offsets:
        return False

    # Check for the sub diagonal
    if -1 not in T.offsets:
        return False

    return True
