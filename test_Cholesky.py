#!/usr/bin/env python
"""This script tests the exported functions in Cholesky.py"""

import numpy as np
import scipy.sparse.linalg as LA
import scipy.sparse as sp
import Cholesky as Chol
import pytest

np.set_printoptions(precision=3)

def test_CholeskyFactorization_Correct():
    """Tests the Cholesky.CholeskyFactorization Method"""
    # tolerance is machine epsilon
    print("-----------------------------------------------------")
    print("Testing the correctness of the Cholesky Factorization")
    print("-----------------------------------------------------")
    tol = np.finfo(np.float64).eps

    n = 10
    A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n,n), dtype="d", format="dia")
    print("Matrix A is\n", A.toarray())
    L = Chol.CholeskyFactorization(A)
    print("Cholesky Factorization yields\n", L.toarray())

    # Calculate relative error:
    rel_error = LA.norm(L*L.T - A, ord=np.inf)/LA.norm(A,ord=np.inf)
    print("Relative Error is ", rel_error)

    # Check if the relative error is less than machine epsilon
    assert rel_error < tol

def test_CholeskyFactorization_Convert():
    print("\n-----------------------------------------------------")
    print("Testing the format conversion")
    print("-----------------------------------------------------")
    tol = np.finfo(np.float64).eps

    n = 10
    A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n,n), dtype="d", format="dok")
    print("Matrix A is\n", A.toarray())
    L = Chol.CholeskyFactorization(A)
    print("Cholesky Factorization yields\n", L.toarray())

    # Calculate relative error:
    rel_error = LA.norm(L*L.T - A, ord=np.inf)/LA.norm(A,ord=np.inf)
    print("Relative Error is ", rel_error)

    # Check if the relative error is less than machine epsilon
    assert rel_error < tol

def test_CholeskyFactorization_Symmetric():
    print("\n-----------------------------------------------------")
    print("Testing that the function throws an error for non-symmetric")
    print("-----------------------------------------------------")
    offsets = np.array([1,0,-1])
    main_diag = 4.0*np.ones(4, dtype="double")
    sup_diag = np.array([0, 1, 2, 1], dtype="double")
    sub_diag = np.ones(4, dtype="double")
    data = np.vstack((sup_diag, main_diag, sub_diag))
    A = sp.dia_matrix((data, offsets), shape=(4,4))
    print("Matrix A is\n", A.toarray())

    with pytest.raises(RuntimeError) as excinfo:
        L = Chol.CholeskyFactorization(A)
    assert "Matrix must be symmetric" in str(excinfo.value)
    print("Function successfully raised error: Matrix must be symmetric")

def test_CholeskyFactorization_Definite():
    print("\n-----------------------------------------------------")
    print("Testing that the function throws an error for indefinite")
    print("-----------------------------------------------------")
    offsets = np.array([1,0,-1])
    main_diag = 4.0*np.ones(4, dtype="double")
    main_diag[2] = 0.5
    sup_diag = np.ones(4, dtype="double")
    data = np.vstack((sup_diag, main_diag, sup_diag))
    A = sp.dia_matrix((data, offsets), shape=(4,4))
    print("Matrix A is\n", A.toarray())

    with pytest.raises(RuntimeError) as excinfo:
        L = Chol.CholeskyFactorization(A)
    assert "Matrix must be positive definite" in str(excinfo.value)
    print("Function successfully raised error: Matrix must be positive definite")

def test_CholeskyFactorization_Tridiagonal():
    print("\n-----------------------------------------------------")
    print("Testing that the function throws an error for non-tridiagonal")
    print("-----------------------------------------------------")
    offsets = np.array([1,0,-1])
    main_diag = 4.0*np.ones(4, dtype="double")
    sup_diag = np.ones(4, dtype="double")
    data = np.vstack((sup_diag, main_diag, sup_diag))
    A = sp.dia_matrix((data, offsets), shape=(4,4)).todok()
    A[1,3] = A[3,1] = 1.0
    A = sp.dia_matrix(A)
    print("Matrix A is\n", A.toarray())

    with pytest.raises(RuntimeError) as excinfo:
        L = Chol.CholeskyFactorization(A)
    assert "Matrix must be tridiagonal" in str(excinfo.value)
    print("Function successfully raised error: Matrix must be tridiagonal")

def test_CholeskyFactorization_Square():
    print("\n-----------------------------------------------------")
    print("Testing that the function throws an error for non-square")
    print("-----------------------------------------------------")
    offsets = np.array([1,0,-1])
    main_diag = 4.0*np.ones(5, dtype="double")
    sup_diag = np.ones(5, dtype="double")
    data = np.vstack((sup_diag, main_diag, sup_diag))
    A = sp.dia_matrix((data, offsets), shape=(4,5))
    print("Matrix A is\n", A.toarray())

    with pytest.raises(RuntimeError) as excinfo:
        L = Chol.CholeskyFactorization(A)
    assert "Matrix must be square" in str(excinfo.value)
    print("Function successfully raised error: Matrix must be square")
