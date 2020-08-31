#!/usr/bin/env python
"""This script tests the exported functions in Cholesky.py"""

import numpy as np
import scipy.sparse.linalg as LA
import scipy.sparse as sp
import Cholesky as Chol
import pytest

def test_CholeskyFactorization():
    """Tests the Cholesky.CholeskyFactorization Method"""
    tol = 1e-8

    n = 10
    ex = np.ones(n, dtype="double")
    data = np.array([-ex, 2*ex, -ex])
    offsets = np.array([1,0,-1])
    A = sp.dia_matrix((data,offsets), shape=(n,n))

    L = Chol.CholeskyFactorization(A)
    assert LA.norm(L*L.T - A, ord=np.inf) < tol

    A = sp.dok_matrix(A)
    L = Chol.CholeskyFactorization(A)
    assert LA.norm(L*L.T - A, ord=np.inf) < tol

    main_diag = 4.0*np.ones(4, dtype="double")
    sup_diag = np.array([0, 1, 2, 1], dtype="double")
    sub_diag = np.ones(4, dtype="double")
    data = np.vstack((sup_diag, main_diag, sub_diag))
    A = sp.dia_matrix((data, offsets), shape=(4,4))

    with pytest.raises(RuntimeError) as excinfo:
        L = Chol.CholeskyFactorization(A)
    assert "Matrix must be symmetric" in str(excinfo.value)

    main_diag = 4.0*np.ones(4, dtype="double")
    main_diag[2] = 0.5
    sup_diag = np.ones(4, dtype="double")
    data = np.vstack((sup_diag, main_diag, sup_diag))
    A = sp.dia_matrix((data, offsets), shape=(4,4))

    with pytest.raises(RuntimeError) as excinfo:
        L = Chol.CholeskyFactorization(A)
    assert "Matrix must be positive definite" in str(excinfo.value)

    main_diag = 4.0*np.ones(4, dtype="double")
    sup_diag = np.ones(4, dtype="double")
    data = np.vstack((sup_diag, main_diag, sup_diag))
    A = sp.dia_matrix((data, offsets), shape=(4,4)).toarray()
    A[1,3] = A[3,1] = 1.0
    A = sp.dia_matrix(A)

    with pytest.raises(RuntimeError) as excinfo:
        L = Chol.CholeskyFactorization(A)
    assert "Matrix must be tridiagonal" in str(excinfo.value)

    main_diag = 4.0*np.ones(5, dtype="double")
    sup_diag = np.ones(5, dtype="double")
    data = np.vstack((sup_diag, main_diag, sup_diag))
    A = sp.dia_matrix((data, offsets), shape=(4,5))

    with pytest.raises(RuntimeError) as excinfo:
        L = Chol.CholeskyFactorization(A)
    assert "Matrix must be square" in str(excinfo.value)
