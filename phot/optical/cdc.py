import numpy as np
from numpy.fft import fftshift, fft, ifft, ifftshift


def cdc(re_x, re_y, power_temp_x, power_temp_y, sampling_rate, beta2, span, L):
    power_temp2_x = np.mean(abs(re_x) ** 2)
    power_temp2_y = np.mean(abs(re_y) ** 2)

    re_x = re_x / (np.sqrt(power_temp2_x / power_temp_x))
    re_y = re_y / (np.sqrt(power_temp2_y / power_temp_y))

    data_length = np.size(re_x)
    omega = np.arange(-sampling_rate / 2, sampling_rate / 2, sampling_rate / data_length)
    omega = np.reshape(omega, (-1, 1))

    re_x_fre = fftshift(fft(re_x)) / np.size(re_x)
    re_y_fre = fftshift(fft(re_y)) / np.size(re_y)

    re_x_fre = re_x_fre * np.exp(1j * (beta2 / 2) * (omega ** 2) * (span * L))
    re_y_fre = re_y_fre * np.exp(1j * (beta2 / 2) * (omega ** 2) * (span * L))

    re_x = ifft(ifftshift(re_x_fre)) * np.size(re_x)
    re_y = ifft(ifftshift(re_y_fre)) * np.size(re_y)

    return re_x, re_y
