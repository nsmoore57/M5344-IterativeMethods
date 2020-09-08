#!/usr/bin/env python
"""This script tests the exported functions in Cholesky.py"""

import numpy as np
import scipy.sparse.linalg as LA
import scipy.sparse as sp
import pytest
import Cholesky as Chol

np.set_printoptions(precision=3)

def _Sub_and_Main_Diag_Only(A):
    """Used to test that the factorization has only a main diagonal and sub diagonal"""
    if len(A.offsets) != 2:
        return False
    if -1 not in A.offsets:
        return False
    if 0 not in A.offsets:
        return False
    return True


def test_CholeskyFactorization_Correct():
    """Tests the Cholesky.CholeskyFactorization for correct factorization"""
    print("-----------------------------------------------------")
    print("Testing the correctness of the Cholesky Factorization")
    print("-----------------------------------------------------")
    # tolerance is machine epsilon
    tol = np.finfo(np.float64).eps

    # Set up A matrix
    n = 10
    A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), dtype=np.double, format="dia")
    print("Matrix A is\n", A.toarray())

    # Run the Cholesky Factorization
    L = Chol.CholeskyFactorization(A)

    # Factorization should be in dia format
    assert sp.isspmatrix_dia(L)

    # Factorization should have have main and sub diagonals
    assert _Sub_and_Main_Diag_Only(L)

    # Output the found factorization
    print("Cholesky Factorization yields\n", L.toarray())

    # Calculate relative error:
    rel_error = LA.norm(L*L.T - A, ord=np.inf)/LA.norm(A, ord=np.inf)
    print("Relative Error is ", rel_error)

    # Check if the relative error is less than machine epsilon
    assert rel_error < tol

def test_CholeskyFactorization_Convert():
    """Test that CholeskyFactorization correctly converts from other formats"""
    print("\n-----------------------------------------------------")
    print("Testing the format conversion")
    print("-----------------------------------------------------")
    # tolerance is machine epsilon
    tol = np.finfo(np.float64).eps

    # Set up A matrix
    n = 10
    A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), dtype=np.double, format="dok")
    print("Matrix A is\n", A.toarray())

    # Run the Cholesky Factorization
    L = Chol.CholeskyFactorization(A)

    # Factorization should be in dia format
    assert sp.isspmatrix_dia(L)

    # Factorization should have have main and sub diagonals
    assert _Sub_and_Main_Diag_Only(L)

    # Output the found factorization
    print("Cholesky Factorization yields\n", L.toarray())

    # Calculate relative error:
    rel_error = LA.norm(L*L.T - A, ord=np.inf)/LA.norm(A, ord=np.inf)
    print("Relative Error is ", rel_error)

    # Check if the relative error is less than machine epsilon
    assert rel_error < tol

def test_CholeskyFactorization_Symmetric():
    """Test that CholeskyFactorization errors for non-symmetric matrices"""
    print("\n-----------------------------------------------------")
    print("Testing that the function throws an error for non-symmetric")
    print("-----------------------------------------------------")

    # Set up A matrix
    offsets = np.array([1, 0, -1])
    main_diag = 4.0*np.ones(4, dtype="double")
    sup_diag = np.array([1, 2, 1], dtype="double")
    sub_diag = np.ones(3, dtype="double")
    A = sp.diags([sup_diag, main_diag, sub_diag], offsets, shape=(4, 4), dtype=np.double, format="dia")
    print("Matrix A is\n", A.toarray())

    # check if function throws the correct error
    with pytest.raises(RuntimeError) as excinfo:
        Chol.CholeskyFactorization(A)
    assert "Matrix must be symmetric" in str(excinfo.value)
    print("Function successfully raised error: Matrix must be symmetric")

def test_CholeskyFactorization_Definite():
    """Test that CholeskyFactorization errors for indefinite matrices"""
    print("\n-----------------------------------------------------")
    print("Testing that the function throws an error for indefinite")
    print("-----------------------------------------------------")

    # Set up A matrix
    offsets = np.array([1, 0, -1])
    main_diag = 4.0*np.ones(4, dtype="double")
    main_diag[2] = 0.5
    sup_diag = np.ones(3, dtype="double")
    A = sp.diags([sup_diag, main_diag, sup_diag], offsets, shape=(4, 4), dtype=np.double, format="dia")
    print("Matrix A is\n", A.toarray())

    # check if function throws the correct error
    with pytest.raises(RuntimeError) as excinfo:
        Chol.CholeskyFactorization(A)
    assert "Matrix must be positive definite" in str(excinfo.value)
    print("Function successfully raised error: Matrix must be positive definite")

def test_CholeskyFactorization_Tridiagonal():
    """Test that CholeskyFactorization errors for non-tridiagonal matrices"""
    print("\n-----------------------------------------------------")
    print("Testing that the function throws an error for non-tridiagonal")
    print("-----------------------------------------------------")

    # Set up A matrix
    offsets = np.array([1, 0, -1])
    main_diag = 4.0*np.ones(4, dtype="double")
    sup_diag = np.ones(3, dtype="double")

    # Build the matrix in dok format so that we can add two more entries
    A = sp.diags([sup_diag, main_diag, sup_diag], offsets, shape=(4, 4), dtype=np.double, format="dok")
    A[1, 3] = A[3, 1] = 1.0

    # Now convert to dia
    A = sp.dia_matrix(A)
    print("Matrix A is\n", A.toarray())

    # check if function throws the correct error
    with pytest.raises(RuntimeError) as excinfo:
        Chol.CholeskyFactorization(A)
    assert "Matrix must be tridiagonal" in str(excinfo.value)
    print("Function successfully raised error: Matrix must be tridiagonal")

def test_CholeskyFactorization_Square():
    """Test that CholeskyFactorization errors for non-square matrices"""
    print("\n-----------------------------------------------------")
    print("Testing that the function throws an error for non-square")
    print("-----------------------------------------------------")

    # Set up A matrix
    offsets = np.array([1, 0, -1])
    main_diag = 4.0*np.ones(4, dtype="double")
    sup_diag = np.ones(4, dtype="double") # Can also be used for sub_diag
    A = sp.diags([sup_diag, main_diag, sup_diag], offsets, shape=(4, 5), dtype="d", format="dia")
    print("Matrix A is\n", A.toarray())

    # check if function throws the correct error
    with pytest.raises(RuntimeError) as excinfo:
        Chol.CholeskyFactorization(A)
    assert "Matrix must be square" in str(excinfo.value)
    print("Function successfully raised error: Matrix must be square")
