#!/usr/bin/env python
"""This script tests the exported functions in Cholesky.py"""

import numpy as np
import scipy.sparse.linalg as LA
import scipy.sparse as sp
import Cholesky as Chol
import pytest

def test_CholeskyFactorization_Correct():
    """Tests the Cholesky.CholeskyFactorization Method"""
    # tolerance is machine epsilon
    tol = np.finfo(np.float64).eps

    n = 10
    ex = np.ones(n, dtype="double")
    data = np.array([-ex, 2*ex, -ex])
    offsets = np.array([1,0,-1])
    A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n,n), dtype="d", format="dia")

    L = Chol.CholeskyFactorization(A)
    # Check if the relative error is less than machine epsilon
    assert LA.norm(L*L.T - A, ord=np.inf)/LA.norm(A,ord=np.inf) < tol

def test_CholeskyFactorization_Convert():
    tol = np.finfo(np.float64).eps

    n = 10
    ex = np.ones(n, dtype="double")
    data = np.array([-ex, 2*ex, -ex])
    offsets = np.array([1,0,-1])
    A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n,n), dtype="d", format="dok")
    L = Chol.CholeskyFactorization(A)
    # Check if the relative error is less than machine epsilon
    assert LA.norm(L*L.T - A, ord=np.inf)/LA.norm(A,ord=np.inf) < tol

def test_CholeskyFactorization_Symmetric():
    offsets = np.array([1,0,-1])
    main_diag = 4.0*np.ones(4, dtype="double")
    sup_diag = np.array([0, 1, 2, 1], dtype="double")
    sub_diag = np.ones(4, dtype="double")
    data = np.vstack((sup_diag, main_diag, sub_diag))
    A = sp.dia_matrix((data, offsets), shape=(4,4))

    with pytest.raises(RuntimeError) as excinfo:
        L = Chol.CholeskyFactorization(A)
    assert "Matrix must be symmetric" in str(excinfo.value)

def test_CholeskyFactorization_Definite():
    offsets = np.array([1,0,-1])
    main_diag = 4.0*np.ones(4, dtype="double")
    main_diag[2] = 0.5
    sup_diag = np.ones(4, dtype="double")
    data = np.vstack((sup_diag, main_diag, sup_diag))
    A = sp.dia_matrix((data, offsets), shape=(4,4))

    with pytest.raises(RuntimeError) as excinfo:
        L = Chol.CholeskyFactorization(A)
    assert "Matrix must be positive definite" in str(excinfo.value)

def test_CholeskyFactorization_Tridiagonal():
    offsets = np.array([1,0,-1])
    main_diag = 4.0*np.ones(4, dtype="double")
    sup_diag = np.ones(4, dtype="double")
    data = np.vstack((sup_diag, main_diag, sup_diag))
    A = sp.dia_matrix((data, offsets), shape=(4,4)).toarray()
    A[1,3] = A[3,1] = 1.0
    A = sp.dia_matrix(A)

    with pytest.raises(RuntimeError) as excinfo:
        L = Chol.CholeskyFactorization(A)
    assert "Matrix must be tridiagonal" in str(excinfo.value)

def test_CholeskyFactorization_Square():
    offsets = np.array([1,0,-1])
    main_diag = 4.0*np.ones(5, dtype="double")
    sup_diag = np.ones(5, dtype="double")
    data = np.vstack((sup_diag, main_diag, sup_diag))
    A = sp.dia_matrix((data, offsets), shape=(4,5))

    with pytest.raises(RuntimeError) as excinfo:
        L = Chol.CholeskyFactorization(A)
    assert "Matrix must be square" in str(excinfo.value)
