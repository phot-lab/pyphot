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
from math import floor, ceil


def cma_rde(input_x_i, input_x_q, input_y_i, input_y_q, num_tap, num_update_cma, ref_power_cma, step_size_cma,
            step_size_rde, samples_per_symbol, bits_per_symbol):
    """
    Function description:  Common CMA with arbitrary tap number and sampling rate
    CMA算法原理可参考：Faruk, Md Saifuddin, and Seb J. Savory. "Digital signal processing for coherent transceivers employing multilevel formats."
    Journal of Lightwave Technology 35.5 (2017): 1125-1141.
    Args:
        input_x_i:
        input_x_q:
        input_y_i:
        input_y_q:
        num_tap: number of FIR taps for CMA, must be an odd number
        num_update_cma: Iteration count for CMA
        ref_power_cma: the modulus^2 for CMA
        step_size_cma: step size for CMA
        step_size_rde:
        samples_per_symbol: oversampling rate
        bits_per_symbol:

    Returns:
        output_x
        output_y
    """
    # Initialize the tap coefficients  对均衡器的抽头系数进行初始化，设置中间的系数为1，其他为0
    initial_weight_xx = np.zeros((num_tap, 1))
    initial_weight_yx = np.zeros((num_tap, 1))
    initial_weight_xy = np.zeros((num_tap, 1))
    initial_weight_yy = np.zeros((num_tap, 1))
    initial_weight_xx[int(floor(num_tap - 1) / 2)] = 1
    initial_weight_yy[int(floor(num_tap - 1) / 2)] = 1

    # Normalization
    norm_factor = np.sqrt(
        2 / np.mean(input_x_i ** 2 + input_x_q ** 2 + input_y_i ** 2 + input_y_q ** 2))  # 对信号进行归一化，加快运算速度
    input_x_i = input_x_i * norm_factor
    input_x_q = input_x_q * norm_factor
    input_y_i = input_y_i * norm_factor
    input_y_q = input_y_q * norm_factor

    # I&Q combined to complex signals
    qpsk_x = input_x_i + 1j * input_x_q  # 对实部虚部进行组合为复数信号
    qpsk_y = input_y_i + 1j * input_y_q

    # 加上CMA预均衡收敛的信号长度
    qpsk_x = np.concatenate((qpsk_x[0:samples_per_symbol * num_update_cma], qpsk_x), axis=0)
    qpsk_y = np.concatenate((qpsk_y[0:samples_per_symbol * num_update_cma], qpsk_y), axis=0)

    # Initialization for CMA
    l_prior = floor((num_tap - 1) / 2)  # 计算抽头的左边界
    l_after = ceil((num_tap - 1) / 2)  # 计算抽头的右边界
    num_length = len(qpsk_x)

    num_output = floor((num_length - num_tap + 1) / samples_per_symbol)
    num_update = num_update_cma + num_output  # 计算全部的信号长度

    w_x = np.zeros((2 * num_tap, num_update + 1), dtype=np.complex128)
    w_y = np.zeros((2 * num_tap, num_update + 1), dtype=np.complex128)

    # 将各个FIR滤波器系数组合
    w_x[:, 0] = np.concatenate((initial_weight_xx, initial_weight_yx), axis=0).ravel()
    w_y[:, 0] = np.concatenate((initial_weight_xy, initial_weight_yy), axis=0).ravel()

    e_x = np.zeros(num_update, dtype=np.complex128)
    e_y = np.zeros(num_update, dtype=np.complex128)

    output_x = np.zeros((num_output, 1), dtype=np.complex128)
    output_y = np.zeros((num_output, 1), dtype=np.complex128)

    # 预均衡 Convergence
    # 采用CMA算法进行抽头系数预收敛

    ind_u = 0
    for ind in range(l_prior, l_prior + 1 + samples_per_symbol * (num_update_cma - 1), samples_per_symbol):
        prior = ind - l_prior - 1 if ind - l_prior - 1 >= 0 else None
        s = np.concatenate(
            (qpsk_x[ind + l_after:prior:-1], qpsk_y[ind + l_after:prior:-1]), axis=0)
        z_x = s.T @ w_x[:, ind_u]
        z_y = s.T @ w_y[:, ind_u]

        e_x[ind_u] = ref_power_cma - np.square(np.abs(z_x))  # 计算CMA的误差函数
        e_y[ind_u] = ref_power_cma - np.square(np.abs(z_y))

        w_x[:, ind_u + 1] = w_x[:, ind_u] + step_size_cma * z_x * np.conj(s) * e_x[ind_u]  # 更新CMA的抽头系数
        w_y[:, ind_u + 1] = w_y[:, ind_u] + step_size_cma * z_y * np.conj(s) * e_y[ind_u]

        ind_u = ind_u + 1

    # 设置不同星座图圆圈下的半径
    r1 = np.sqrt(2)
    r2 = np.sqrt(10)
    r3 = np.sqrt(18)
    r4 = np.sqrt(26)
    r5 = np.sqrt(34)
    r6 = np.sqrt(50)
    r7 = np.sqrt(58)
    r8 = np.sqrt(74)
    r9 = np.sqrt(98)

    # Realtime equalization
    # 采用RDE算法进行正式均衡
    ind_a = 0
    for ind in range(l_prior, l_prior + 1 + samples_per_symbol * (num_output - 1), samples_per_symbol):
        prior = ind - l_prior - 1 if ind - l_prior - 1 >= 0 else None
        s = np.concatenate((qpsk_x[ind + l_after:prior:-1], qpsk_y[ind + l_after:prior:-1]), axis=0)
        z_x = s.T @ w_x[:, ind_u]
        z_y = s.T @ w_y[:, ind_u]

        # 计算不同调制格式下的误差函数
        if bits_per_symbol == 2:  # QPSK
            # X偏振下的误差函数
            e_x[ind_u] = r1 - abs(z_x) ** 2
            # Y偏振下的误差函数
            e_y[ind_u] = r1 - abs(z_y) ** 2
        elif bits_per_symbol == 4:  # 16QAM
            # X
            if abs(z_x) ** 2 <= 0.5 * (r1 + r2):
                e_x[ind_u] = r1 - abs(z_x) ** 2
            elif abs(z_x) ** 2 <= 0.5 * (r2 + r3):
                e_x[ind_u] = r2 - abs(z_x) ** 2
            else:
                e_x[ind_u] = r3 - abs(z_x) ** 2
            # Y
            if abs(z_y) ** 2 <= 0.5 * (r1 + r2):
                e_y[ind_u] = r1 - abs(z_y) ** 2
            elif abs(z_y) ** 2 <= 0.5 * (r2 + r3):
                e_y[ind_u] = r2 - abs(z_y) ** 2
            else:
                e_y[ind_u] = r3 - abs(z_y) ** 2
        elif bits_per_symbol == 5:  # 32QAM
            # X
            if abs(z_x) ** 2 <= 0.5 * (r1 + r2):
                e_x[ind_u] = r1 - abs(z_x) ** 2
            elif abs(z_x) ** 2 <= 0.5 * (r2 + r3):
                e_x[ind_u] = r2 - abs(z_x) ** 2
            elif abs(z_x) ** 2 <= 0.5 * (r3 + r4):
                e_x[ind_u] = r3 - abs(z_x) ** 2
            elif abs(z_x) ** 2 <= 0.5 * (r4 + r5):
                e_x[ind_u] = r4 - abs(z_x) ** 2
            else:
                e_x[ind_u] = r5 - abs(z_x) ** 2
            # Y
            if abs(z_y) ** 2 <= 0.5 * (r1 + r2):
                e_y[ind_u] = r1 - abs(z_y) ** 2
            elif abs(z_y) ** 2 <= 0.5 * (r2 + r3):
                e_y[ind_u] = r2 - abs(z_y) ** 2
            elif abs(z_y) ** 2 <= 0.5 * (r3 + r4):
                e_y[ind_u] = r3 - abs(z_y) ** 2
            elif abs(z_y) ** 2 <= 0.5 * (r4 + r5):
                e_y[ind_u] = r4 - abs(z_y) ** 2
            else:
                e_y[ind_u] = r5 - abs(z_y) ** 2

        # 利用RDE算法对抽头系数进行更新
        w_x[:, ind_u + 1] = w_x[:, ind_u] + step_size_rde * e_x[ind_u] * z_x * np.conj(s)  # 更新CMA的抽头系数
        w_y[:, ind_u + 1] = w_y[:, ind_u] + step_size_rde * e_y[ind_u] * z_y * np.conj(s)
        ind_u = ind_u + 1

        output_x[ind_a] = z_x
        output_y[ind_a] = z_y
        ind_a = ind_a + 1

    # 祛除添加的预均衡信号，并去掉前后部分均衡效果不好的信号
    output_x = output_x[num_update_cma + 1999:-2000]
    output_y = output_y[num_update_cma + 1999:-2000]

    return output_x, output_y
