import numpy as np
from scipy.linalg import lapack
from helper_functions import *
from math import log, sqrt
from time import time
#import pprofile

def dormqr(side, transpose, qr, tau, a, overwrite_c=0):
    lwork = lapack.dormqr(side, transpose, qr, tau, a, -1, overwrite_c)[1][0]
    return lapack.dormqr(side, transpose, qr, tau, a, lwork, overwrite_c)

@compiler_decorator
def power_method_iteration(d, n2, V, ntries, maxiter, eigtol, gradtol, ftol):

    # C_n from Lemma 4.7
    if n2 <= 4:
      cn = sqrt(2*(n2-1)/n2)
    else:
      cn = (2-sqrt(2))*sqrt(n2)


    for tries in range(ntries):

        # Initialize Ak
        Ak = np.random.randn(d)
        Ak = Ak / norm(Ak)

        for iter in range(maxiter):

            # Calculate power of Ak
            Apow = khatri_rao_power(Ak.reshape((-1, 1)), n2 - 1).reshape(-1)

            # Calculate contraction of V with x ^ (n2 - 1)
            VAk = np.dot(Apow, V).reshape((d, -1))
            Ak_new = np.dot(VAk, np.dot(Ak, VAk))
            #VAk = dgemv(1., V_, Apow, trans=1).reshape((d, -1))
            #Ak_new = dgemv(1., VAk, dgemv(1., VAk, Ak, trans=1))

            f = dot(Ak_new, Ak)

            # Determine optimal shift
            # Due to numerical errors f may be greater than 1
            fcl = np.maximum(np.minimum(f, 1.), .5)
            clambda = sqrt(fcl * (1 - fcl))
            shift = cn * clambda

            if f < ftol:
                # Xk was a very bad initialization
                # This happens very rarely
                # Initialize it again at random
                Ak = np.random.randn(d)
                Ak = Ak / norm(Ak)
                continue

            Ak_new = Ak_new + shift * Ak
            Ak_new = Ak_new / norm(Ak_new)

            err = norm(Ak - Ak_new)
            Ak = Ak_new
            if err < gradtol:
                break

        if 1 - f < eigtol:
            break
        elif tries == 0:
            f_ = f
            Ak_ = Ak
        elif f > f_:
            f_ = f
            Ak_ = Ak
        else:
            Ak = Ak_

    return Ak

def subspace_power_method(T, d=None, n=None, r=None, **kwargs):

    if d is None:
        d = T.shape[0]
    if n is None:
        n = round(log(T.size) / log(d))

    assert n % 2 == 0 and n > 0, f'"Argument n={n} is not even an even positive integer';

    pos = lambda x: x>0
    isbool = lambda x: isinstance(x, bool)

    opts = option_parser(kwargs,
                         ('maxiter', 5000, pos),
                         ('ntries', 3, pos),
                         ('gradtol', 1e-14, pos),
                         ('eigtol', 1e-8, pos),
                         ('ftol', 1e-2 / sqrt(d), pos),
                         ('w_out', True, isbool))

    n2 = n // 2
    dn2 = d ** n2

    matT = T.reshape(dn2, dn2)

    symind, findsym, symindscale = symmetric_indices(d, n2)

    symindscale = np.sqrt(symindscale)
    findsym = findsym.flatten()
    symind = symind[::-1,:].T @ (d ** np.arange(n2))

    sym_matT = symindscale.reshape(1, -1) * matT[symind][:, symind] * symindscale.reshape(-1, 1)

    D, symV = eig2(sym_matT)

    if r is None:
        r = D.shape[0] - np.searchsorted(D[::-1], opts.eigtol)

    D = D[:r]
    V = (symV[:, :r] / symindscale.reshape(-1, 1))[findsym, :]

    D1 = np.diagflat(1. / D).T

    A = np.empty((d, r))
    w = np.empty(r)

    for k in range(r):

        Ak = power_method_iteration(d, n2, V.reshape((d ** (n2 -1) , -1)),
                                    opts.ntries, opts.maxiter, opts.eigtol,
                                    opts.gradtol, opts.ftol) #V[:,k:].copy()

        # Calculate power of Ak
        Apow = khatri_rao_power(Ak.reshape(-1, 1), n2)

        # Calculate projection of Xpow in subspace
        alpha = (Apow.T @ V).T

        D1alpha = D1 @ alpha
        A[:, k] = Ak
        w[k] = 1. / (alpha.T @ D1alpha)

        if k:
            # Calculate the new matrix D and the new subspace
            # Use Householder reflection to update V and D
            qr, tau, work, info = lapack.dgeqrf(D1alpha, overwrite_a=1)

            D1, work, info = dormqr('R', 'T', qr, tau, D1, overwrite_c=1)
            D1, work, info = dormqr('L', 'N', qr, tau, D1, overwrite_c=1)
            D1 = D1[1:, 1:]

            V, work, info = dormqr('R', 'T', qr, tau, V, overwrite_c=1)
            V = V[:, 1:]

    if opts.w_out:
        return A, w
    else:
        return A * w.reshape(1, -1) ** (1./n)

if __name__ == '__main__':
    class empty_with():
        def __enter__(self):
            pass

        def __exit__(self, *args, **kwargs):
            pass

  #  profiler = pprofile.Profile()
    #with empty_with():
    d = 20
    r = 120
    n = 4
    A = np.random.randn(d, r)
    #A = A / np.linalg.norm(A, axis=0)
    T = generate_lowrank_tensor(A, n=n)

    start = time()
    A_ = subspace_power_method(T, w_out=False)
    print(time()-start)

    d = 20
    r = 120
    n = 4
    A = np.random.randn(d, r)
    #A = A / np.linalg.norm(A, axis=0)
    T = generate_lowrank_tensor(A, n=n)

    start = time()
    A_ = subspace_power_method(T, w_out=False)
    print(time()-start)

    d = 45
    r = 800
    n = 4
    A = np.random.randn(d, r)
    # A = A / np.linalg.norm(A, axis=0)
    T = generate_lowrank_tensor(A, n=n)

    start = time()
    A_ = subspace_power_method(T, w_out=False)
    print(time() - start)

    '''with profiler():
        d = 40
        r = 600
        n = 4
        A = np.random.randn(d, r)
        # A = A / np.linalg.norm(A, axis=0)
        T = generate_lowrank_tensor(A, n=n)

        start = time()
        A_ = subspace_power_method(T, w_out=False)
        print(time() - start)

    profiler.dump_stats("profiler_stats.txt")'''