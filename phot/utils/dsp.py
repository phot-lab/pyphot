"""
Copyright (c) 2022 Beijing Jiaotong University
PhotLab is licensed under [Open Source License].
You can use this software according to the terms and conditions of the [Open Source License].
You may obtain a copy of [Open Source License] at: [https://open.source.license/]

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.

See the [Open Source License] for more details.

Author: Chunyu Li
Created: 2022/9/6
Supported by: National Key Research and Development Program of China
"""

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
