import time
from numba import njit, prange
from jax.lax import fori_loop


def normal_func():
    a = 10
    for i in range(1000000000):
        a = 32 * 32 + 324 * 432


@njit
def numba_func():
    a = 10
    for i in prange(1000000000):
        a = 32 * 32 + 324 * 432


def jax_func():
    def func(i, a):
        a = 32 * 32 + 324 * 432
        return a

    a = 10
    a = fori_loop(0, 1000000000, func, a)


if __name__ == '__main__':
    start = time.time()
    numba_func()
    print('Time: {}s'.format(time.time() - start))
