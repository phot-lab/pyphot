"""
calc stands for calculate, as its name, this module includes
functions for arithmetic calculation
"""

import fractions
import numpy as np
from scipy.stats import unitary_group


def fast_exp(x):
    return np.exp(1j * x)


def rat(number):
    """
    Simulates the rat function in MATLAB
    :param number: given number to be converted into fraction form
    :return: numerator and denominator
    """
    fraction = fractions.Fraction(number).limit_denominator()
    return fraction.numerator, fraction.denominator


def dec2bin(array, num_bits):
    array = array.astype(np.uint8)  # np.unpackbits func requires uint8 datatype
    array = np.fliplr(np.unpackbits(array, axis=1, bitorder='little', count=int(num_bits)))
    return array.astype(np.int8)


def rand_unitary(dim):
    """
    Generate random unitary matrix
    :param dim: dimension of generated matrix
    :return: matrix
    """
    return unitary_group.rvs(dim=dim)


def n_mod(a, n):
    """
    NMOD N-modulus of an integer.
    Y=NMOD(A,N) reduces the integer A into 1->N, mod N.

    E.g. N=8.

    A   ... -2 -1 0 1 2 3 4 5 6 7 8 9 10 ...
    Y   ...  6  7 8 1 2 3 4 5 6 7 8 1 2  ...
    :param a:
    :param n:
    :return:
    """
    return np.mod(a - 1 - n, n) + 1
