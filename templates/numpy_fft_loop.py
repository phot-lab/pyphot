import time
import numpy as np


def func_numpy(a):
    a = np.fft.fft(a)
    a = np.fft.ifft(a)
    return a


if __name__ == '__main__':
    upper = 100000
    a = np.random.random(4324)

    start = time.time()
    for i in range(upper):
        a = func_numpy(a)
    print('Numpy FFT loop time: {}s'.format(time.time() - start))
