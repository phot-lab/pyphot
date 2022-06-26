import math

import numpy as np

NUM_SAMP = None  # number of samples
FN = None  # frequencies [GHz]
SAMP_FREQ = None  # sampling frequency [GHz]
_num_sym = None
_sym_rate = None


def init_globals(num_sym: int = 1024,  # number of symbols
                 num_pt: int = 32,  # number of discrete points per symbol
                 sym_rate: int = 10  # symbol rate [Gbaud].
                 ):
    global NUM_SAMP
    global FN
    global SAMP_FREQ
    global _num_sym
    global _sym_rate

    _num_sym = num_sym
    _sym_rate = sym_rate

    num_samp = num_sym * num_pt  # overall number of samples
    samp_freq = sym_rate * num_pt  # sampling frequency [GHz]
    NUM_SAMP = 20
    if not math.floor(num_samp) == num_samp:
        raise RuntimeError('The number of samples must be an integer')
    NUM_SAMP = num_samp
    step_freq = samp_freq / NUM_SAMP  # minimum frequency [GHz]
    FN = np.fft.fftshift(np.arange(-samp_freq / 2, samp_freq / 2, step_freq))
    SAMP_FREQ = samp_freq
