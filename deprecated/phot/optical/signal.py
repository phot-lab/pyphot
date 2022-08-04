import numpy as np
import math


def up_sample(array, n):
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


def down_sample(array: np.ndarray):
    rows, cols = np.shape(array)
    result = np.zeros((rows, 1), dtype=complex)
    for i in range(rows):
        result[i][0] = max(array[i, :], key=lambda x: np.abs(x))
    return result


def my_filter(fileter_type: str, freq, bandwidth, param=None):
    """

    :param fileter_type:
    :param freq:
    :param bandwidth:
    :param param: An optional parameter used by some filters.
    :return:
    """
    np.reshape(freq, (-1, 1), order='F')
    x = freq / bandwidth  # frequency normalized to the bandwidth
    fileter_type = fileter_type.lower()
    if fileter_type == "movavg":
        hf = np.sinc(x)  # Short term integrator
    elif fileter_type == "gauss":
        hf = np.exp(-0.5 * math.log(2) * x * x)  # Gaussian
    elif fileter_type == "rc1":
        hf = 1 / (1 + 1j * x)  # RC filter 1st order == Butterworth 1st order
    elif fileter_type == "rc2":
        # RC filter 2nd order
        hf = 1 / np.square((1 + 1j * np.sqrt((np.sqrt(2) - 1) * x)))  # |Hf|^2=0.5 for x=+-1
    elif fileter_type == "rootrc":
        # Root Raised Cosine
        if param is None:
            raise RuntimeError("Missing filter roll-off.")
        x = x / 2  # convert two-sided bandwidth in low-pass
        if param < 0 or param > 1:
            raise RuntimeError("It must be 0<=roll-off<=1")
        hf = np.zeros(x.shape)
        hf[np.abs(x) <= 0.5 * (1 - param)] = 1
        ii = (0.5 * (1 - param) < np.abs(x)) & (np.abs(x) <= 0.5 * (1 + param))
        hf[ii] = np.sqrt(0.5 * (1 + np.cos(math.pi / param * (np.abs(x[ii]) - 0.5 * (1 - param)))))
    else:
        raise RuntimeError("the filter ftype does not exist.")
    return hf


def interp(array, factor):
    """
    Interpolate, i.e. up-sample a given 1D vector by a specific interpolation factor.
    :param array: 1D data array
    :param factor: factor for interpolation (must be integer)
    :return: interpolated 1D array by a given factor
    """
    x = np.arange(array.size)
    store = np.linspace(x[0], x[-1], np.size(x) * factor)
    return np.interp(store, x, array)


# author - Mathuranathan Viswanathan (gaussianwaves.com)
# This code is part of the book Digital Modulations using Python
# https://www.gaussianwaves.com/2015/06/how-to-generate-awgn-noise-in-matlaboctave-without-using-in-built-awgn-function/
def awgn(s, snr_db, factor=1):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        snr_db : desired signal to noise ratio (expressed in dB) for the received signal
        factor : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
    gamma = 10 ** (snr_db / 10)  # SNR to linear scale
    if s.ndim == 1:  # if s is single dimensional vector
        P = factor * sum(abs(s) ** 2) / len(s)  # Actual power in the vector
    else:  # multi-dimensional signals like MFSK
        P = factor * sum(sum(abs(s) ** 2)) / len(s)  # if s is a matrix [MxN]
    n0 = P / gamma  # Find the noise spectral density
    if np.isrealobj(s):  # check if input is real/complex object type
        n = np.sqrt(n0 / 2) * np.random.standard_normal(s.shape)  # computed noise
    else:
        n = np.sqrt(n0 / 2) * (np.random.standard_normal(s.shape) + 1j * np.random.standard_normal(s.shape))
    r = s + n  # received signal
    return r
