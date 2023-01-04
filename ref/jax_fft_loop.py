import jax
import jax.numpy as jnp
import time
import numpy as np
from jax.lax import fori_loop


def func(i, a):
    a = jax.numpy.fft.fft(a)
    a = jax.numpy.fft.ifft(a)
    return a


if __name__ == '__main__':
    upper = 100000
    ar = np.random.random(4324)
    ar = jax.numpy.array(ar)
    ar = jnp.asarray(ar, dtype=jnp.complex64)

    start = time.time()
    ar = fori_loop(0, upper, func, ar)
    print('Jax FFT loop Time: {}s'.format(time.time() - start))
