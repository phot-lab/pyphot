import sys

import numpy as np
from numpy import sqrt
from math import floor, ceil


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

    # 第一步 测试角

    signal_with_angle = np.zeros((total_number, num_test_angle), dtype=complex)
    for ind in range(total_number):
        for point in range(num_test_angle):
            # [-0.5pi,0.5pi]:由于QAM信号的PI/2的旋转对称性， 使用不同测试角对信号进行预补偿
            signal_with_angle[ind][point] = complex_signal[ind][0] * np.exp(
                1j * (point / num_test_angle - 0.5) * np.pi / 2)
    # 得到了一个用测试角旋转后的矩阵

    # 判决
    signal_decided = np.zeros((total_number, num_test_angle), dtype=complex)
    for poinm in range(num_test_angle):
        signal_decided[:, poinm] = decision_hybrid_qam(signal_with_angle[:, poinm], bits_per_symbol)  # 对预补偿后的信号进行判决

    # 得到距离矩阵
    distance_matrix = signal_with_angle - signal_decided
    distance_matrix_square = np.abs(distance_matrix) ** 2  # 求得判决后的信号与预补偿信号的差值功率

    error = np.zeros((total_number, num_test_angle))  # 得到误差矩

    for poinn in range(num_test_angle):
        for inn in range(total_number):
            block_start = inn - ceil(block_size / 2) + 1  # 计算滑动窗口左边界
            block_end = inn + floor(block_size / 2)  # 计算滑动窗口右边界
            if block_start < 0:
                block_start = 0
            if block_end > total_number:
                block_end = total_number
            error[inn][poinn] = np.sum(distance_matrix_square[block_start:block_end, poinn])  # 计算不同测试角下，块长内误差和

    # 通过最小距离来得到估计得相位值
    decision_fist_stage = np.zeros((total_number, 1), dtype=complex)
    signal_first_stage = np.zeros((total_number, 1), dtype=complex)
    min_result = np.zeros((total_number, 2))

    for inr in range(total_number):
        # 后面error每一行所得的最小值的列号，就是第几个测试角是正确估计的相位噪声所对应的测试角的序号
        min_result[inr][0] = np.min(error[inr, :])
        min_result[inr][1] = np.argmin(error[inr, :])
        decision_fist_stage[inr][0] = signal_decided[inr][int(min_result[inr][1])]  # 这是理想的点
        signal_first_stage[inr][0] = signal_with_angle[inr][int(min_result[inr][1])]  # 这是对应的角度旋转后的符号

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


def decision_hybrid_qam(input_signal, m):
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

    # 对星座点进行归一化，因为接收信号归一化了，这里进行对应
    if m == 2:
        constellation_hy = constellation_4qam / sqrt(2)
    elif m == 3:
        constellation_hy = constellation_8qam / sqrt(3 + sqrt(3))
    elif m == 4:
        constellation_hy = constellation_16qam / sqrt(10)
    elif m == 5:
        constellation_hy = constellation_32qam / sqrt(20)
    else:
        raise RuntimeError('Unknown bits number')

    # 判断每个信号与哪个星座点的欧式距离最小，则判决这个信号为距离最小的星座点
    input_signal = input_signal.reshape((1, -1))
    m, n = input_signal.shape
    out = np.zeros((m, n), dtype=complex)
    for i in range(m):
        for j in range(n):
            min_index = np.argmin(np.abs(constellation_hy - input_signal[i][j]))
            out[i][j] = constellation_hy[min_index]

    return out
