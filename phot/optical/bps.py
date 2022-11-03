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
from numpy import sqrt
from math import floor, ceil
from numba import njit, prange


def bps_hybrid_qam(input_i, input_q, num_test_angle, block_size, bits_per_symbol):
    """
    此处显示有关此函数的摘要
    采用BPS算法，具体参考文献为：
    Pfau, Timo, Sebastian Hoffmann, and Reinhold Noé.
    "Hardware-efficient coherent digital receiver concept with feedforward carrier recovery for  M-QAM constellations."
    Journal of Lightwave Technology 27.8 (2009): 989-999.
    Args:
        input_i:
        input_q:
        num_test_angle:
        block_size:
        bits_per_symbol:

    Returns:

    """
    # 调整信号矩阵为1列
    input_i = np.reshape(input_i, (-1, 1))
    input_q = np.reshape(input_q, (-1, 1))

    norm_factor = np.sqrt(np.mean(np.abs(input_i) ** 2 + np.abs(input_q) ** 2))  # 进行归一化
    input_i = input_i / norm_factor
    input_q = input_q / norm_factor

    complex_signal = input_i + 1j * input_q  # 组合为复信号
    total_number = len(complex_signal)  # 计算信号长度

    # 得到了一个用测试角旋转后的矩阵
    signal_with_angle = _angle_matrix(total_number, num_test_angle, complex_signal)

    constellation_hy = _init_constellation(bits_per_symbol)

    # 判决
    signal_decided = np.zeros((total_number, num_test_angle), dtype=complex)
    for poinm in range(num_test_angle):
        signal_decided[:, poinm] = decision_hybrid_qam(np.reshape(signal_with_angle[:, poinm], (1, -1)),
                                                       constellation_hy)  # 对预补偿后的信号进行判决

    # 得到距离矩阵
    distance_matrix = signal_with_angle - signal_decided
    distance_matrix_square = np.abs(distance_matrix) ** 2  # 求得判决后的信号与预补偿信号的差值功率

    error = _error_matrix(total_number, num_test_angle, block_size, distance_matrix_square)

    min_result = _calculate_min_result(total_number, error)

    phase = np.zeros((total_number, 1))
    for i in range(total_number):
        phase[i][0] = (min_result[i][1] / num_test_angle - 0.5) * np.pi / 2  # 得到测试角序号对应的测试角

    phase_est_uwapx_compress = np.zeros((len(phase), 1))
    c = np.zeros((len(phase), 1))

    # phase unwrap 用来判断相位是不是属于其它象限，如果是的话将偏移角度变化，对估计的相位进行展开，解决周期滑动
    for i in range(1, len(phase)):
        if abs(phase[i][0] - phase[i - 1][0]) > np.pi / 4:  # 判断前后相位是否相差大于pi/4，大于的话需要加pi/2
            c[i][0] = c[i - 1][0] - np.sign(phase[i][0] - phase[i - 1][0]) * np.pi / 2
        else:
            c[i][0] = c[i - 1][0]
        phase_est_uwapx_compress[i][0] = phase[i][0] + c[i][0]  # 将前后相位进行展开

    output_signal = complex_signal * np.exp(1j * phase_est_uwapx_compress)  # 对信号进行相位噪声补偿
    return output_signal, phase_est_uwapx_compress


@njit
def _calculate_min_result(total_number, error):
    # 通过最小距离来得到估计得相位值
    # decision_fist_stage = np.zeros((total_number, 1), dtype=np.complex128)
    # signal_first_stage = np.zeros((total_number, 1), dtype=np.complex128)
    min_result = np.zeros((total_number, 2))

    for inr in range(total_number):
        # 后面error每一行所得的最小值的列号，就是第几个测试角是正确估计的相位噪声所对应的测试角的序号
        min_result[inr][0] = np.min(error[inr, :])
        min_result[inr][1] = np.argmin(error[inr, :])
        # decision_fist_stage[inr][0] = signal_decided[inr][int(min_result[inr][1])]  # 这是理想的点
        # signal_first_stage[inr][0] = signal_with_angle[inr][int(min_result[inr][1])]  # 这是对应的角度旋转后的符号
    return min_result


@njit(parallel=True)
def _angle_matrix(total_number, num_test_angle, complex_signal):
    # 第一步 测试角
    signal_with_angle = np.zeros((total_number, num_test_angle), dtype=np.complex128)
    for ind in prange(total_number):
        for point in prange(num_test_angle):
            # [-0.5pi,0.5pi]:由于QAM信号的PI/2的旋转对称性， 使用不同测试角对信号进行预补偿
            signal_with_angle[ind][point] = complex_signal[ind][0] * np.exp(
                1j * (point / num_test_angle - 0.5) * np.pi / 2)
    # 得到了一个用测试角旋转后的矩阵

    return signal_with_angle


def _init_constellation(bits_per_symbol):
    # 列出各个调制格式下的原始星座点
    constellation_4qam = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j])

    constellation_8qam = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j, -1 - sqrt(3), 1 + sqrt(3), (-1 - sqrt(3)) * 1j,
                                   (1 + sqrt(3)) * 1j])

    constellation_16qam = np.array([-3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j, -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
                                    1 - 3j, 1 - 1j, 1 + 1j, 1 + 3j, 3 - 3j, 3 - 1j, 3 + 1j, 3 + 3j])

    constellation_32qam = np.array(
        [5 + 3j, 5 + 1j, 5 - 1j, 5 - 3j, 3 + 5j, 3 + 3j, 3 + 1j, 3 - 1j, 3 - 3j, 3 - 5j, 1 + 5j,
         1 + 3j, 1 + 1j, 1 - 1j, 1 - 3j, 1 - 5j, -1 + 5j, -1 + 3j, -1 + 1j, -1 - 1j, -1 - 3j, -1 - 5j,
         -3 + 5j, -3 + 3j, -3 + 1j, -3 - 1j, -3 - 3j, -3 - 5j, -5 + 3j, -5 + 1j, -5 - 1j, -5 - 3j])

    constellation_64qam = np.array(
        [7 + 7j, 7 + 5j, 7 + 3j, 7 + 1j, 7 - 1j, 7 - 3j, 7 - 5j, 7 - 7j, 5 + 7j, 5 + 5j, 5 + 3j, 5 + 1j, 5 - 1j, 5 - 3j,
         5 - 5j, 5 - 7j, 3 + 7j, 3 + 5j, 3 + 3j, 3 + 1j, 3 - 1j, 3 - 3j, 3 - 5j, 3 - 7j, 1 + 7j, 1 + 5j, 1 + 3j, 1 + 1j,
         1 - 1j, 1 - 3j, 1 - 5j, 1 - 7j, -1 + 7j, -1 + 5j, -1 + 3j, -1 + 1j, -1 - 1j, -1 - 3j, -1 - 5j, -1 - 7j,
         -3 + 7j, -3 + 5j, -3 + 3j, -3 + 1j, -3 - 1j, -3 - 3j, -3 - 5j, -3 - 7j, -5 + 7j, -5 + 5j, -5 + 3j, -5 + 1j,
         -5 - 1j, -5 - 3j, -5 - 5j, -5 - 7j, -7 + 7j, -7 + 5j, -7 + 3j, -7 + 1j, -7 - 1j, -7 - 3j, -7 - 5j, -7 - 7j])

    # 对星座点进行归一化，因为接收信号归一化了，这里进行对应
    if bits_per_symbol == 2:
        constellation_hy = constellation_4qam / sqrt(2)
    elif bits_per_symbol == 3:
        constellation_hy = constellation_8qam / sqrt(3 + sqrt(3))
    elif bits_per_symbol == 4:
        constellation_hy = constellation_16qam / sqrt(10)
    elif bits_per_symbol == 5:
        constellation_hy = constellation_32qam / sqrt(20)
    elif bits_per_symbol == 6:
        constellation_hy = constellation_64qam / sqrt(42)
    else:
        raise RuntimeError('Unknown bits number')

    return constellation_hy


@njit(parallel=True)
def _error_matrix(total_number, num_test_angle, block_size, distance_matrix_square):
    error = np.zeros((total_number, num_test_angle))  # 得到误差矩

    for poinn in prange(num_test_angle):
        for inn in prange(total_number):
            block_start = inn - ceil(block_size / 2) + 1  # 计算滑动窗口左边界
            block_end = inn + floor(block_size / 2)  # 计算滑动窗口右边界
            if block_start < 0:
                block_start = 0
            if block_end > total_number:
                block_end = total_number
            error[inn][poinn] = np.sum(distance_matrix_square[block_start:block_end, poinn])  # 计算不同测试角下，块长内误差和
    return error


@njit(parallel=True)
def decision_hybrid_qam(input_signal, constellation_hy):
    # 判断每个信号与哪个星座点的欧式距离最小，则判决这个信号为距离最小的星座点

    m, n = input_signal.shape

    # if not utils.is_gpu(constellation_hy):
    #     constellation_hy = utils.to_gpu(constellation_hy)

    # input_signal_temp = np.ravel(input_signal, order='F')
    # with Pool(processes=8) as p:
    #     out = p.map(_compute_distance, input_signal_temp)
    # out = np.reshape(out, (m, n), order='F')

    out = np.zeros((m, n), dtype=np.complex128)
    for i in prange(m):
        for j in prange(n):
            min_index = np.argmin(np.abs(constellation_hy - input_signal[i][j]))
            out[i][j] = constellation_hy[min_index]

    return out
