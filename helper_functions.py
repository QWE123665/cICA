import numpy as np
from argparse import Namespace
from compiler_options import compiler_decorator, dot, norm
import timeit
from math import factorial
from itertools import permutations


def option_parser(kwargs, *args):

    opts = {}
    for option_name, default_val, test in args:
        val = kwargs.get(option_name, default_val)
        assert isinstance(option_name, str), f'{repr(option_name)} is not a valid string'
        if test is not None:
            assert test(val), f'Test failed for option {option_name}'
        opts[option_name] = val

    return Namespace(**opts)

@compiler_decorator
def khatri_rao_product(A, B):
    k = A.shape[1]

    AB = A.reshape((-1,1,k)) * B.reshape((1,-1,k))

    return AB.reshape((-1, k))

@compiler_decorator
def khatri_rao_power(A, n):
    if n == 1:
        return A
    elif n == 2:
        return khatri_rao_product(A, A)
    else:
        n2 = n // 2
        An = khatri_rao_power(A, n2)
        if n % 2:
            An = khatri_rao_product(An, khatri_rao_product(An, A))
        else:
            An = khatri_rao_product(An, An)
        return An


def tucker_product(T: np.ndarray, M: np.ndarray, n: int = None) -> np.ndarray:
    if n is None:
        n = len(T.shape)

    d, k = M.shape

    T = T.reshape([k] * n + [-1])

    for i in range(n):
        T = np.dot(M, T)

    try:
        T = T.squeeze(axis=n)
    except ValueError:
        pass

    return T



def symmetric_indices(d, n):

    # Code for calculating subsets of fixed dimension adapted from Paul Panzer's code
    # see nump2 in https://stackoverflow.com/questions/42138681/faster-numpy-solution-instead-of-itertools-combinations/42202157#42202157

    symind = np.ones((n, d), dtype=int)
    symind[0] = np.arange(d)
    nperm = np.full(d, factorial(n))
    last_equal = np.ones(d)
    for j in range(1, n):
        reps = d - symind[j-1]
        symind = np.repeat(symind, reps, axis=1)
        ind = np.cumsum(reps)
        symind[j, ind[:-1]] = 1-reps[1:]
        symind[j, 0] = 0
        symind[j] = np.cumsum(symind[j], out=symind[j])
        new_last_equal = np.ones(symind.shape[1])
        new_last_equal[ind[:-1]] = last_equal[1:] + 1
        new_last_equal[0] = j + 1
        last_equal = new_last_equal
        nperm = np.repeat(nperm, reps) / last_equal

    dnsym = symind.shape[1]

    findsym = np.empty(d ** n, dtype=int)
    for perm in permutations(range(n)):
        findsym[symind[perm, :].T  @ (d ** np.arange(n))] = np.arange(dnsym)
    findsym = findsym.reshape([d] * n)

    return symind, findsym, nperm


def generate_lowrank_tensor(X, n, lbd = None):

    d, k = X.shape
    if lbd is None:
        lbd = np.ones((1, k))

    assert n == abs(round(n - 1)) + 1, f'The input {n} is not a positive integer.'

    if n == 1:
        return np.dot(X, lbd)

    n2 = n // 2
    X_pow = khatri_rao_power(X, n2)
    if n % 2:
        X_pow1 = khatri_rao_product(X_pow, X)
    else:
        X_pow1 = X_pow
    X_pow1 = X_pow1 * lbd.reshape(1, k)
    out = np.dot(X_pow1, X_pow.T)
    return out.reshape([d]*n)


def generate_lowrank_tensor_general(W, A, n=None):
    if n is None:
        n = len(W[0].shape)

    for i, (Wi, Ai) in enumerate(zip(W, A)):
        Ti = tucker_product(Wi, Ai, n)
        if i:
            T += Ti
        else:
            T = Ti

    return T


def eig2(a):
    D, V = np.linalg.eigh(a)
    #ind = (-D).argsort(axis=-1)
    ind = (-abs(D)).argsort(axis=-1)
    D = np.take_along_axis(D, ind, axis=-1)
    V = np.take_along_axis(V, np.expand_dims(ind,-2), axis=-1)
    return D, V



if __name__ == '__main__':
    T1 = generate_lowrank_tensor(np.ones((3, 5)), lbd=np.arange(5), n=4)
    T2 = tucker_product(T1, np.random.randn(2, 3))
    T3 = generate_lowrank_tensor_general((np.random.randn(3,3,3), np.random.randn(2,2,2)),
                                        (np.random.randn(5,3), np.random.randn(5,2)))
    n = 100
    #print(timeit.timeit("a = set_comb_replacement(50,5)", setup="from __main__ import set_comb_replacement",number=n)/n)
    symind, findsym, nperm = symmetric_indices(3,3)

    symind, findsym, symindscale = symmetric_indices(3, 3)

    symindscale = np.sqrt(symindscale)
    findsym = findsym.flatten()
    symind = symind.T @ (3 ** np.arange(3))

    print('A')


