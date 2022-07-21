import numpy as np


def read_matrix(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    arr = np.zeros((len(lines), 1), dtype=complex)
    for idx, line in enumerate(lines):
        line = line.replace('i', 'j')
        arr[idx][0] = complex(line)
    return arr
