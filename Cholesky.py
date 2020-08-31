#!/usr/bin/python
"""Module for Cholesky Methods"""
import numpy as np
import scipy.sparse as sp

def CholeskyFactorization(A):
    """
    Runs a Cholesky Factorization on matrix A.
    Matrix A must be tridiagonal.
    """

    # Check if the matrix is in dia_matrix format, and convert otherwise
    if type(A) == sp.dia_matrix:
        T = A
    else:
        T = sp.dia_matrix(A)

    # Check the dimensions
    (m,n) = T.shape
    if m != n:
        raise RuntimeError("Matrix must be square")

    # Check if the matrix is tridiagonal
    if not(_isTridiagonal(T)):
        raise RuntimeError("Matrix must be tridiagonal")

    # Check for symmetry
    if not all(T.diagonal(-1) == T.diagonal(1)):
        raise RuntimeError("Matrix must be symmetric")

    # Make copies of the necessary diagonals of A to prevent changing A
    L_main_diag = T.diagonal(0).copy()
    L_sub_diag = T.diagonal(-1).copy()

    # Build L column by column
    for i in range(n):
        # If it's the first column, don't need to subtract anything
        if i == 0:
            L_main_diag[i] = L_main_diag[i]
        else:
            L_main_diag[i] -= L_sub_diag[i-1]*L_sub_diag[i-1]

        # Check if matrix is indefinite here to prevent complex arithmetic cost
        if L_main_diag[i] < 0:
            raise RuntimeError("Matrix must be positive definite")

        L_main_diag[i] = np.sqrt(L_main_diag[i])

        # Subdiagonal is one element shorter than main diagonal so can't assign at i == n-1
        if i < n-1:
            L_sub_diag[i] /= L_main_diag[i]

    L = sp.dia_matrix((n,n), dtype="d")
    L.setdiag(L_main_diag,k=0)
    L.setdiag(L_sub_diag,k=-1)
    return L



def _isTridiagonal(L):
    """
    Checks to see whether L is tridiagonal.
    L must be in scipy.sparse.dia_matrix form
    """
    # First check if there's 3 diagonals
    if len(L.offsets) != 3:
        return False

    # Check for the main diagonal
    if 0 not in L.offsets:
        return False

    # Check for a super diagonal
    if 1 not in L.offsets:
        return False

    # Check for a sub diagonal
    if -1 not in L.offsets:
        return False

    return True
