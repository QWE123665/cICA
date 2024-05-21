import numpy as np

try:
    from numba import njit
    compiler_decorator = njit
    NUMBA_COMPILER = True

except ModuleNotFoundError:
    def compiler_decorator(fun):
        return fun

    NUMBA_COMPILER = True

BLAS_DOT = False
if not NUMBA_COMPILER:
    try:
        from scipy.linalg.blas import dnrm2 as norm
        from scipy.linalg.blas import ddot as dot

        BLAS_DOT = True
    except ModuleNotFoundError:
        pass


if not BLAS_DOT:

    dot = np.dot

    @compiler_decorator
    def norm(v):
        return np.sqrt(np.dot(v, v))
