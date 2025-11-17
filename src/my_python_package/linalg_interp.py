import numpy as np
def gauss_iter_solve(A,b,x0=None,tol=1e-8,alg='seidel'):
    """calculate to estimate x in Ax=b iteratively
    Parameters
    ----------
    A : array_like, the coefficient matrix.
    b : array_like, right-hand-side vector/resulting vector
    x0 : array_like, the initial guess of x defult=None, same shape as b
    tol: float, relative error tolerance, defult=1e-8
    alg: str, flag for the algorithm to be used 
    ----------
    Returns
    x in np.ndarray, same shape as b
    """
    # raise an error if alg does not fit the criteria
    alg = alg.strip().lower()
    if not(alg=="seidel" or alg=="jacobi"):
        raise ValueError(f"arg 'alg' have to be 'seidel' or 'jacobi', Got {alg}")

    # Convert to numpy arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # Check A is 2D
    if A.ndim != 2:
        raise ValueError(f"A must be 2D. Got ndim={A.ndim}")
    
    n_rows, n_cols = A.shape
    
    # Check A is square
    if n_rows != n_cols:
        raise ValueError(f"A must be square. Got shape={A.shape}")
    
    # Check b is 1D or column vector and length matches
    if b.ndim == 1:
        if b.shape[0] != n_rows:
            raise ValueError(f"Length of b ({b.shape[0]}) does not match number of rows in A ({n_rows})")
    elif b.ndim == 2:
        if b.shape != (n_rows, 1):
            raise ValueError(f"b must be shape ({n_rows},) or ({n_rows}, 1). Got {b.shape}")
    else:
        raise ValueError(f"b must be 1D or 2D column vector. Got ndim={b.ndim}")
    
    # Check x0 if provided
    if x0 is not None:
        x0 = np.array(x0, dtype=float)
        if x0.ndim != 1 or x0.shape[0] != n_cols:
            raise ValueError(f"x0 must be 1D vector of length {n_cols}. Got shape {x0.shape}")
    x = x0.copy() if x0 is not None else np.zeros(n_cols)
        
    # Iterative loop
    max_iter = 100000
    for iteration in range(max_iter):
        x_new = np.zeros_like(x) # To same shape
        for i in range(n_cols):
            s = 0 # reset s
            for j in range(n_cols):
                if j != i:
                    if alg == 'jacobi':
                        s += A[i, j] * x[j]      # previous iteration
                    else:  # Gauss-Seidel
                        s += A[i, j] * x_new[j] if j < i else A[i, j] * x[j]
            x_new[i] = (b[i] - s) / A[i, i]
    
        # Check relative error
        rel_error = np.linalg.norm(x_new - x, ord=np.inf) / np.linalg.norm(x_new, ord=np.inf)
        if rel_error < tol:
            return x_new
        
        x = x_new

    # Max iterations reached
    print("RuntimeWarning: Maximum iterations reached without convergence")
    return x

import numpy as np

def spline_function(xd, yd, order=3):
    """
    Generate a spline interpolation function.
    
    Parameters
    ----------
    xd : array_like, independent variable, must be increasing
    yd : array_like, dependent variable, same length as xd
    order : int, 1 (linear), 2 (quadratic), 3 (cubic), default=3
    
    Returns
    -------
    f : function, callable that takes scalar or array of x values
    """
    # Convert to numpy arrays
    xd = np.array(xd, dtype=float).flatten()
    yd = np.array(yd, dtype=float).flatten()

    # Error checks
    if xd.shape[0] != yd.shape[0]:
        raise ValueError(f"xd and yd must have same length, got {xd.shape[0]} and {yd.shape[0]}")
    if len(np.unique(xd)) != xd.shape[0]:
        raise ValueError("xd contains repeated values")
    if not np.all(np.diff(xd) > 0): # We can use this insted of np.sort()
        raise ValueError("xd must be strictly increasing")
    if order not in [1, 2, 3]:
        raise ValueError("order must be 1, 2, or 3")

    n = len(xd)
    h = np.diff(xd)  # intervals between points

    # Linear spline coefficients
    if order == 1:
        slopes = np.diff(yd) / h

        def f(x_val):
            x_val = np.asarray(x_val, dtype=float)
            if np.any(x_val < xd[0]) or np.any(x_val > xd[-1]):
                raise ValueError(f"x out of bounds: xmin={xd[0]}, xmax={xd[-1]}")
            # Find interval index
            idx = np.searchsorted(xd, x_val) - 1
            idx[idx < 0] = 0  # handle edge case
            return yd[idx] + slopes[idx] * (x_val - xd[idx])
        return f

    # Quadratic spline (natural boundary: second derivative at first point = 0)
    elif order == 2:
        # Solve for c_i coefficients (second derivative)
        A = np.zeros((n, n))
        b = np.zeros(n)
        A[0, 0] = 1  # natural BC: S0'' = 0
        b[0] = 0
        for i in range(1, n-1):
            A[i, i-1] = h[i-1]
            A[i, i] = 2*(h[i-1] + h[i])
            A[i, i+1] = h[i]
            b[i] = 3*((yd[i+1]-yd[i])/h[i] - (yd[i]-yd[i-1])/h[i-1])
        A[-1, -1] = 1  # natural BC at end
        b[-1] = 0
        c = np.linalg.solve(A, b)

        # Coefficients for each interval: a + b*(x-xi) + c*(x-xi)^2
        a = yd[:-1]
        b_coef = (yd[1:] - yd[:-1])/h - h*(2*c[:-1] + c[1:])/3
        c_coef = c[:-1]

        def f(x_val):
            x_val = np.asarray(x_val, dtype=float)
            if np.any(x_val < xd[0]) or np.any(x_val > xd[-1]):
                raise ValueError(f"x out of bounds: xmin={xd[0]}, xmax={xd[-1]}")
            idx = np.searchsorted(xd, x_val) - 1
            idx[idx < 0] = 0
            dx = x_val - xd[idx]
            return a[idx] + b_coef[idx]*dx + c_coef[idx]*dx**2
        return f

    # Cubic spline (natural)
    else:
        # Solve for M_i = second derivatives
        A = np.zeros((n, n))
        rhs = np.zeros(n)
        A[0, 0] = 1
        A[-1, -1] = 1
        for i in range(1, n-1):
            A[i, i-1] = h[i-1]
            A[i, i] = 2*(h[i-1] + h[i])
            A[i, i+1] = h[i]
            rhs[i] = 3*((yd[i+1]-yd[i])/h[i] - (yd[i]-yd[i-1])/h[i-1])
        M = np.linalg.solve(A, rhs)

        # Compute coefficients for each interval
        a = yd[:-1]
        b_coef = (yd[1:] - yd[:-1])/h - h*(2*M[:-1] + M[1:])/3
        c_coef = M[:-1]
        d_coef = (M[1:] - M[:-1])/(3*h)

        def f(x_val):
            x_val = np.asarray(x_val, dtype=float)
            if np.any(x_val < xd[0]) or np.any(x_val > xd[-1]):
                raise ValueError(f"x out of bounds: xmin={xd[0]}, xmax={xd[-1]}")
            idx = np.searchsorted(xd, x_val) - 1
            idx[idx < 0] = 0
            dx = x_val - xd[idx]
            return a[idx] + b_coef[idx]*dx + c_coef[idx]*dx**2 + d_coef[idx]*dx**3
        return f

