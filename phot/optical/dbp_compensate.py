import numpy as np
from numpy.fft import fftshift, fft, ifft


def dbp_compensate(re_x, re_y, power_temp_x, power_temp_y, sampling_rate, span, num_steps, beta2, delta_z, gamma):
    data_length = np.size(re_x)

    omega = np.arange(-sampling_rate / 2, sampling_rate / 2, sampling_rate / data_length)
    omega = np.reshape(omega, (-1, 1))

    # 信号功率
    power_temp2_x = np.mean(np.abs(re_x) ** 2)
    power_temp2_y = np.mean(np.abs(re_y) ** 2)

    re_x = re_x / (np.sqrt(power_temp2_x / power_temp_x))
    re_y = re_y / (np.sqrt(power_temp2_y / power_temp_y))

    data_fft_x = fftshift(fft(re_x, axis=0), axes=0) / np.size(re_x)  # 将数据变到频率并将零值移到原点
    data_fft_y = fftshift(fft(re_y, axis=0), axes=0) / np.size(re_y)

    for i in range(span):
        for j in range(num_steps):
            # 1/2 光纤色散
            # X 偏振
            data_fft_x = data_fft_x * np.exp(1j * (beta2 / 2) * (omega ** 2) * (delta_z / 2))
            # Y 偏振
            data_fft_y = data_fft_y * np.exp(1j * (beta2 / 2) * (omega ** 2) * (delta_z / 2))

            # 光纤非线性
            # X 偏振
            data_t_x = ifft(fftshift(data_fft_x, axes=0), axis=0) * np.size(data_fft_x)
            # Y 偏振
            data_t_y = ifft(fftshift(data_fft_y, axes=0), axis=0) * np.size(data_fft_y)
            # X 偏振
            data_t_x = data_t_x * np.exp(-8 / 9 * 1j * gamma * ((abs(data_t_y) ** 2) + (abs(data_t_x) ** 2)) * delta_z)
            # Y 偏振
            data_t_y = data_t_y * np.exp(-8 / 9 * 1j * gamma * ((abs(data_t_y) ** 2) + (abs(data_t_x) ** 2)) * delta_z)

            data_fft_x = fftshift(fft(data_t_x, axis=0), axes=0) / np.size(data_t_x)
            data_fft_y = fftshift(fft(data_t_y, axis=0), axes=0) / np.size(data_t_y)

            # 1/2 光纤色散
            # X 偏振
            data_fft_x = data_fft_x * np.exp(1j * (beta2 / 2) * (omega ** 2) * (delta_z / 2))
            # Y 偏振
            data_fft_y = data_fft_y * np.exp(1j * (beta2 / 2) * (omega ** 2) * (delta_z / 2))

    re_x = ifft(fftshift(data_fft_x, axes=0), axis=0) * np.size(data_fft_x)
    re_y = ifft(fftshift(data_fft_y, axes=0), axis=0) * np.size(data_fft_y)

    return re_x, re_y
