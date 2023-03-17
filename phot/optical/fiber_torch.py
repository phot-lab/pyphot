import numpy as np
import torch
from torch.fft import fftshift, fft, ifft, ifftshift
import math


def fiber_torch(
    signals,
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
    with torch.no_grad():
        data_x = signals[0]
        data_y = signals[1]

        x_length = np.size(data_x)
        y_length = np.size(data_y)

        data_x = torch.from_numpy(data_x)
        data_y = torch.from_numpy(data_y)
        
        data_x = data_x.to("cuda")
        data_y = data_y.to("cuda")

        data_x_normal = data_x / torch.sqrt(torch.mean(abs(data_x) ** 2))
        data_y_normal = data_y / torch.sqrt(torch.mean(abs(data_y) ** 2))

        p1 = torch.mean(abs(data_x_normal) ** 2)
        p2 = torch.mean(abs(data_y_normal) ** 2)

        # 信号功率
        power_dbm_set = -6
        power_mw = 10 ** (power_dbm_set / 10)
        power_w = power_mw / 1000

        power_origin_x = torch.mean(torch.abs(data_x_normal) ** 2)
        power_origin_y = torch.mean(torch.abs(data_y_normal) ** 2)

        data_x = data_x_normal / torch.sqrt(power_origin_x / power_w)
        data_y = data_y_normal / torch.sqrt(power_origin_y / power_w)

        power_temp_x = torch.mean(torch.abs(data_x) ** 2)
        power_temp_y = torch.mean(torch.abs(data_y) ** 2)

        """ SSFM - Modeling """

        omega = np.arange(-sampling_rate / 2, sampling_rate / 2, sampling_rate / x_length)
        omega = np.reshape(omega, (-1, 1))

        omega = torch.from_numpy(omega)
        omega = omega.to("cuda")

        data_fft_x = fftshift(fft(data_x, dim=0), dim=0) / x_length  # 将数据变到频率并将零值移到原点
        data_fft_y = fftshift(fft(data_y, dim=0), dim=0) / y_length

        for i in range(span):
            for j in range(num_steps):
                # 1/2 信号衰减
                data_fft_x = data_fft_x * math.exp(-alpha * (delta_z / 2))
                data_fft_y = data_fft_y * math.exp(-alpha * (delta_z / 2))

                # 1/2 光纤色散
                # X 偏振
                data_fft_x = data_fft_x * torch.exp(-1j * (beta2 / 2) * (omega**2) * (delta_z / 2))
                # Y 偏振
                data_fft_y = data_fft_y * torch.exp(-1j * (beta2 / 2) * (omega**2) * (delta_z / 2))

                # 光纤非线性
                # X 偏振
                data_t_x = ifft(ifftshift(data_fft_x, dim=0), dim=0) * data_fft_x.size(dim=-1)
                # Y 偏振
                data_t_y = ifft(ifftshift(data_fft_y, dim=0), dim=0) * data_fft_y.size(dim=-1)

                # X 偏振
                data_t_x = data_t_x * torch.exp(
                    8 / 9 * 1j * gamma * ((abs(data_t_y) ** 2) + (abs(data_t_x) ** 2)) * delta_z
                )
                # Y 偏振
                data_t_y = data_t_y * torch.exp(
                    8 / 9 * 1j * gamma * ((abs(data_t_y) ** 2) + (abs(data_t_x) ** 2)) * delta_z
                )

                data_fft_x = fftshift(fft(data_t_x, dim=0), dim=0) / data_t_x.size(dim=-1)
                data_fft_y = fftshift(fft(data_t_y, dim=0), dim=0) / data_t_y.size(dim=-1)

                # 1/2 光纤色散
                # X 偏振
                data_fft_x = data_fft_x * torch.exp(-1j * (beta2 / 2) * (omega**2) * (delta_z / 2))
                # Y 偏振
                data_fft_y = data_fft_y * torch.exp(-1j * (beta2 / 2) * (omega**2) * (delta_z / 2))

                # 1/2 信号衰减
                data_fft_x = data_fft_x * math.exp(-alpha * (delta_z / 2))
                data_fft_y = data_fft_y * math.exp(-alpha * (delta_z / 2))

            # == EDFA ==
            data_fft_x = data_fft_x * math.exp(alpha * L)
            data_fft_y = data_fft_y * math.exp(alpha * L)

        # step-2: Perturbation Theory
        # step-3: Neural Network

        tx_signal_x = ifft(ifftshift(data_fft_x, dim=0), dim=0) * data_fft_x.size(dim=-1)
        tx_signal_y = ifft(ifftshift(data_fft_y, dim=0), dim=0) * data_fft_y.size(dim=-1)

        tx_signal_x = tx_signal_x.cpu()
        tx_signal_y = tx_signal_y.cpu()
        tx_signal_x = tx_signal_x.numpy()
        tx_signal_y = tx_signal_y.numpy()

        power_temp_x = power_temp_x.cpu()
        power_temp_y = power_temp_y.cpu()
        power_temp_x = power_temp_x.numpy()
        power_temp_y = power_temp_y.numpy()

    return [tx_signal_x, tx_signal_y], [power_temp_x, power_temp_y]
