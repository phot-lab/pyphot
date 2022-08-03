import numpy as np
from numba import jit


# https://stackoverflow.com/questions/15983986/convert-quantiz-function-to-python
@jit(nopython=True)
def quantize(signal, partitions, codebook):
    indices = np.zeros(len(signal))
    quanta = np.zeros(len(signal))
    for i, datum in enumerate(signal):
        index = 0
        while index < len(partitions) and datum > partitions[index]:
            index += 1
        indices[i] = index
        quanta[i] = codebook[index]
    return indices, quanta


def upsample(array, n):
    if isinstance(n, float):
        if n.is_integer():
            n = int(n)
        else:
            raise RuntimeError('Up-sample coefficient must be integer')
    rows, cols = array.shape
    result = np.zeros((rows * n, cols), dtype=complex)
    for i in range(rows):
        for j in range(cols):
            result[i * n][j] = array[i][j]
    return result


def volts_to_decibel(x):
    return 20 * np.log10(x)
