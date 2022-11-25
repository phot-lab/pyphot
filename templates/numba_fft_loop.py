import jax
import jax.numpy as jnp
import time
import numpy as np
from jax.lax import fori_loop
from numba import njit, types, objmode, prange


@njit()
def fft_helper(x):
    with objmode(y='complex128[:]'):
        y = np.fft.fft(x)
    return y


@njit()
def ifft_helper(x):
    with objmode(y='complex128[:]'):
        y = np.fft.ifft(x)
    return y


@njit
def numba_func(a):
    for i in prange(100000):
        a = fft_helper(a)
        a = ifft_helper(a)
    return a


if __name__ == '__main__':
    a = np.zeros((4324,), dtype=np.complex128)
    start = time.time()
    numba_func(a)
    print('Numba FFT loop time: {}s'.format(time.time() - start))
