import math
import numpy as np
from deprecated.phot.utils import mathutils
from .format import get_format_info


def seq2samp(seq, mod_format):
    format_info = get_format_info(mod_format)
    num_bits = math.log2(format_info.digit)
    rows, cols = np.shape(seq)  # cols == 1: symbol. cols == 2: bits
    if mod_format == "randn":
        return seq
    seq = mathutils.dec2bin(seq, num_bits)

    # From now on, pat is a binary matrix
    m = math.pow(2, cols)  # constellation size

    if format_info.family == "ook":
        level = 2 * seq  # average energy: 1
    elif mod_format == "bpsk" or mod_format == "dpsk" or mod_format == "psbt" or (mod_format == "psk" and m == 2):
        level = 2 * seq - 1  # average energy: 1
    elif mod_format == "qpsk" or mod_format == "dqpsk":
        level = 2 * seq - 1  # drive iqmodulator with QPSK
        level = (level[:, 0] + level[:, 1] * 1j) / math.sqrt(2)  # average energy: 1
        level = np.reshape(level, (-1, 1))
    else:
        raise RuntimeError("Unknown modulation format")

    return level
