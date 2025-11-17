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
    max_iter = 10000
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

def spline_function(xd, yd, order):
    """Multiply two numbers or arrays.
    Parameters
    ----------
    x : int or float or array_like
    The first number to multiply.
    y : int or float or array_like
    The second number to multiply.
    Returns
    -------
    int or float or array_like
    The (element-wise) product of x and y.
    """
    return None

