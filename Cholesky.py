#!/usr/bin/python
"""Module for Cholesky Methods"""
import numpy as np
import scipy.sparse as sp

# Define Exceptions
class CholeskyError(Exception):
    """Base class for other exceptions"""
    pass

class NonSymmetricError(CholeskyError):
    """For errors when the input matrix is non-symmetric"""
    pass

class IndefiniteErros(CholeskyError):
    """For errors when the input matrix is indefinite"""
    pass

class NonTridiagonalError(CholeskyError):
    """For errors when the input matrix is not tridiagonal"""
    pass

class SizeError(CholeskyError):
    """For errors when the input matrix is non-square"""
    pass


def CholeskyFactorization(A):
    """
    Runs a Cholesky Factorization on matrix A.
    Matrix A must be tridiagonal.
    """

    # Check if the matrix is in dia_matrix format, and convert otherwise
    if type(A) == sp.dia_matrix:
        L = A.copy()
    else:
        L = sp.dia_matrix(A)

    # Check the dimensions
    (m,n) = L.shape
    if m != n:
        raise RuntimeError("Matrix must be square")

    # Check if the matrix is tridiagonal
    if not(_isTridiagonal(L)):
        raise RuntimeError("Matrix must be tridiagonal")

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
