import unittest
import numpy as np
from my_python_package.linalg_interp import *
from scipy.interpolate import UnivariateSpline

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
        np.testing.assert_allclose(x_jacobi, x_expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(x_seidel, x_expected, rtol=1e-6, atol=1e-6)

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
        
        # Check that AA_inv ≈ I, where assert_allclose for better compairson
        np.testing.assert_allclose(A @ X_jacobi, I, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(A @ X_seidel, I, rtol=1e-6, atol=1e-6)

class TestSplineFunction(unittest.TestCase):

    def setUp(self):
        # Example data points
        self.x = np.linspace(0, 5, 6)  # 6 points
        self.linear_y = 2*self.x + 1
        self.quad_y = self.x**2 - 3*self.x + 2
        self.cubic_y = self.x**3 - 2*self.x**2 + self.x - 5
        self.exp_y = np.exp(self.x)

    def test_exact_recovery(self):
        # Linear data: should be exactly recovered by order=1,2,3
        f1 = spline_function(self.x, self.linear_y, order=1)
        f2 = spline_function(self.x, self.linear_y, order=2)
        f3 = spline_function(self.x, self.linear_y, order=3)
        np.testing.assert_allclose(f1(self.x), self.linear_y, rtol=1e-1)
        np.testing.assert_allclose(f2(self.x), self.linear_y, rtol=1e-1)
        np.testing.assert_allclose(f3(self.x), self.linear_y, rtol=1e-1)

        # Quadratic data: exactly recovered by order=2 and 3, not necessarily by 1
        f1_quad = spline_function(self.x, self.quad_y, order=1)
        f2_quad = spline_function(self.x, self.quad_y, order=2)
        f3_quad = spline_function(self.x, self.quad_y, order=3)
        np.testing.assert_allclose(f2_quad(self.x), self.quad_y, rtol=1e-1)
        np.testing.assert_allclose(f3_quad(self.x), self.quad_y, rtol=1e-1)
        with self.assertRaises(AssertionError):
            # Linear spline may not exactly match quadratic data
            np.testing.assert_allclose(f1_quad(self.x), self.quad_y, rtol=1e-1)

        # Cubic data: exactly recovered by order=3
        f3_cubic = spline_function(self.x, self.cubic_y, order=3)
        np.testing.assert_allclose(f3_cubic(self.x), self.cubic_y, rtol=1e-1)

    def test_comparison_with_univariate_spline(self):
        # Use cubic spline for comparison
        f_cubic = spline_function(self.x, self.exp_y, order=3)
        uspline = UnivariateSpline(self.x, self.exp_y, k=3, s=0, ext='raise')
        x_test = np.linspace(self.x[0], self.x[-1], 50)
        np.testing.assert_allclose(f_cubic(x_test), uspline(x_test), rtol=1e-1)

        # Another example: higher-order polynomial
        poly_y = 3*self.x**5 - 2*self.x**3 + self.x - 7
        f_cubic_poly = spline_function(self.x, poly_y, order=3)
        uspline_poly = UnivariateSpline(self.x, poly_y, k=3, s=0, ext='raise')
        x_test_poly = np.linspace(self.x[0], self.x[-1], 50)
        np.testing.assert_allclose(f_cubic_poly(x_test_poly), uspline_poly(x_test_poly), rtol=1e-1)


if __name__ == '__main__':
    unittest.main()


