import unittest
import numpy as np
from my_python_package.linalg_interp import *
class TestGaussIterSolve(unittest.TestCase):
    
    def test_single_rhs_vector(self):
        # Example system
        A = np.array([[4, 1, 2],
                      [3, 5, 1],
                      [1, 1, 3]], dtype=float)
        b = np.array([4, 7, 3], dtype=float)
        
        # Solve using numpy.linalg.solve
        x_expected = np.linalg.solve(A, b)
        
        # Solve using our iterative solver
        x_jacobi = gauss_iter_solve(A, b, alg='jacobi')
        x_seidel = gauss_iter_solve(A, b, alg='seidel')
        
        # Check that the solutions are close
        np.testing.assert_allclose(x_jacobi, x_expected, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(x_seidel, x_expected, rtol=1e-8, atol=1e-10)

    def test_inverse_matrix(self):
        # Example square matrix
        A = np.array([[2, 1],
                      [5, 7]], dtype=float)
        n = A.shape[0]
        I = np.eye(n)
        
        # Solve for X such that AX = I (each column separately)
        X_jacobi = np.zeros_like(A)
        X_seidel = np.zeros_like(A)
        
        for i in range(n):
            e = I[:, i]
            X_jacobi[:, i] = gauss_iter_solve(A, e, alg='jacobi')
            X_seidel[:, i] = gauss_iter_solve(A, e, alg='seidel')
        
        # Check that AA_inv ≈ I, where assert_allclose for better compareson
        np.testing.assert_allclose(A @ X_jacobi, I, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(A @ X_seidel, I, rtol=1e-8, atol=1e-10)

if __name__ == '__main__':
    unittest.main()


