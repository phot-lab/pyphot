import numpy as np
import math


def up_sample(array, n):
    rows, cols = array.shape
    result = np.zeros((rows * n, cols), dtype=complex)
    for i in range(rows):
        for j in range(cols):
            result[i * n][j] = array[i][j]
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
        ii = 0.5 * (1 - param) < np.abs(x) <= 0.5 * (1 + param)
        hf[ii] = np.sqrt(0.5 * (1 + np.cos(math.pi / param * (np.abs(x[ii])) - 0.5 * (1 - param))))
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
