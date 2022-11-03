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
from scipy.special import erfcinv


def ber_count(received_sequence, transmit_sequence):
    num_transmission = len(transmit_sequence)  # 计算信号长度
    num_receive = len(received_sequence)

    # 调整信号为1列
    received_sequence = np.reshape(received_sequence, (-1, 1))
    transmit_sequence = np.reshape(transmit_sequence, (-1, 1))

    # 计算发射端信号和接收端信号的长度差，标记为需要丢掉的长度
    num_abandon = num_transmission - num_receive

    total_error_sequence = np.zeros((num_abandon + 1, 1))

    for ind in range(num_abandon + 1):  # 这个循环是在计算每种情况下的错误码元长度
        temp = received_sequence[0:num_receive] - transmit_sequence[ind:num_receive + ind]
        total_error_sequence[ind] = len(np.argwhere(temp != 0))

    total_error = np.min(total_error_sequence)  # 定义错误码元数最少的为信号的错误码元数

    total_bits = num_receive
    ber = total_error / total_bits  # 用错误码元数除以全部的符号数，就是误码率
    q = np.sqrt(2) * erfcinv(2 * ber)  # 利用公式利用BER计算出Q因子
    q_db = 20 * np.log10(q)  # 调整单位为dB

    return ber, q_db


def ber_estimate(tx_scm_matrix, rx_scm_matrix, bits_per_symbol):
    # Count the number of error symbols.
    length_symbols = np.size(tx_scm_matrix)
    error_sym = 0
    delta_tr_matrix = rx_scm_matrix - tx_scm_matrix
    delta_iq = np.concatenate((np.abs(np.real(delta_tr_matrix)), np.abs(np.imag(delta_tr_matrix))), axis=1)
    for sym_num in range(length_symbols):
        if delta_iq[sym_num][0] > 1:
            error_sym = error_sym + 1
        if delta_iq[sym_num][1] > 1:
            error_sym = error_sym + 1
    # Estimate the BER value
    ber = error_sym / (2 * length_symbols) / (bits_per_symbol / 2)
    q = np.sqrt(2) * erfcinv(2 * ber)  # 利用公式利用BER计算出Q因子
    q_db = 20 * np.log10(q)  # 调整单位为dB

    return ber, q_db
