import numpy as np
from numpy.fft import fftshift, fft, ifft, ifftshift


def optical_fiber_channel(
    tx_signal_x: np.ndarray,
    tx_signal_y: np.ndarray,
    sampling_rate: float,
    span,
    num_steps,
    beta2,
    delta_z,
    gamma,
    alpha,
    L,
):
    """

    Args:
        tx_signal_x: 输入信号x
        tx_signal_y: 输入信号y
        sampling_rate:
        span: 一个跨度
        num_steps:
        beta2:
        delta_z: 单步步长
        gamma:

    Returns:

    """
    # step-1: Digital Backpropagation

    data_x = tx_signal_x
    data_y = tx_signal_y

    data_x_normal = data_x / np.sqrt(np.mean(abs(data_x) ** 2))
    data_y_normal = data_y / np.sqrt(np.mean(abs(data_y) ** 2))

    p1 = np.mean(abs(data_x_normal) ** 2)
    p2 = np.mean(abs(data_y_normal) ** 2)

    data_length = np.size(data_x)

    # 信号功率
    power_dbm_set = -6
    power_mw = 10 ** (power_dbm_set / 10)
    power_w = power_mw / 1000

    power_origin_x = np.mean(np.abs(data_x_normal) ** 2)
    power_origin_y = np.mean(np.abs(data_y_normal) ** 2)

    data_x = data_x_normal / np.sqrt(power_origin_x / power_w)
    data_y = data_y_normal / np.sqrt(power_origin_y / power_w)

    power_temp_x = np.mean(np.abs(data_x) ** 2)
    power_temp_y = np.mean(np.abs(data_y) ** 2)

    """ SSFM - Modeling """

    omega = np.arange(-sampling_rate / 2, sampling_rate / 2, sampling_rate / data_length)
    omega = np.reshape(omega, (-1, 1))

    data_fft_x = fftshift(fft(data_x, axis=0), axes=0) / np.size(data_x)  # 将数据变到频率并将零值移到原点
    data_fft_y = fftshift(fft(data_y, axis=0), axes=0) / np.size(data_y)

    for i in range(span):
        for j in range(num_steps):
            # 1/2 信号衰减
            data_fft_x = data_fft_x * np.exp(-alpha * (delta_z / 2))
            data_fft_y = data_fft_y * np.exp(-alpha * (delta_z / 2))

            # 1/2 光纤色散
            # X 偏振
            data_fft_x = data_fft_x * np.exp(-1j * (beta2 / 2) * (omega**2) * (delta_z / 2))
            # Y 偏振
            data_fft_y = data_fft_y * np.exp(-1j * (beta2 / 2) * (omega**2) * (delta_z / 2))

            # 光纤非线性
            # X 偏振
            data_t_x = ifft(ifftshift(data_fft_x, axes=0), axis=0) * np.size(data_fft_x)
            # Y 偏振
            data_t_y = ifft(ifftshift(data_fft_y, axes=0), axis=0) * np.size(data_fft_y)

            # X 偏振
            data_t_x = data_t_x * np.exp(8 / 9 * 1j * gamma * ((abs(data_t_y) ** 2) + (abs(data_t_x) ** 2)) * delta_z)
            # Y 偏振
            data_t_y = data_t_y * np.exp(8 / 9 * 1j * gamma * ((abs(data_t_y) ** 2) + (abs(data_t_x) ** 2)) * delta_z)

            data_fft_x = fftshift(fft(data_t_x, axis=0), axes=0) / np.size(data_t_x)
            data_fft_y = fftshift(fft(data_t_y, axis=0), axes=0) / np.size(data_t_y)

            # 1/2 光纤色散
            # X 偏振
            data_fft_x = data_fft_x * np.exp(-1j * (beta2 / 2) * (omega**2) * (delta_z / 2))
            # Y 偏振
            data_fft_y = data_fft_y * np.exp(-1j * (beta2 / 2) * (omega**2) * (delta_z / 2))

            # 1/2 信号衰减
            data_fft_x = data_fft_x * np.exp(-alpha * (delta_z / 2))
            data_fft_y = data_fft_y * np.exp(-alpha * (delta_z / 2))

        # == EDFA ==
        data_fft_x = data_fft_x * np.exp(alpha * L)
        data_fft_y = data_fft_y * np.exp(alpha * L)

    # step-2: Perturbation Theory
    # step-3: Neural Network

    tx_signal_x = ifft(ifftshift(data_fft_x, axes=0), axis=0) * np.size(data_fft_x)
    tx_signal_y = ifft(ifftshift(data_fft_y, axes=0), axis=0) * np.size(data_fft_y)

    return tx_signal_x, tx_signal_y, power_temp_x, power_temp_y
