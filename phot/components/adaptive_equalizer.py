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
from ..optical import cma_rde
from ..utils import plot_scatter
from phot import logger


def adaptive_equalize(
    signals,
    num_tap,
    cma_convergence,
    ref_power_cma,
    step_size_cma,
    step_size_rde,
    up_sampling_factor,
    bits_per_symbol,
    total_baud,
):
    signal_x = signals[0]
    signal_y = signals[1]
    input_x_i = np.real(signal_x)  # 求出接收端X偏振的实部信号
    input_x_q = np.imag(signal_x)  # 求出接收端X偏振的虚部信号
    input_y_i = np.real(signal_y)  # 求出接收端Y偏振的实部信号
    input_y_q = np.imag(signal_y)  # 求出接收端Y偏振的实部信号

    # 对信号采用CMA-RDE进行自适应均衡
    equalization_matrix_x, equalization_matrix_y = cma_rde(
        input_x_i,
        input_x_q,
        input_y_i,
        input_y_q,
        num_tap,
        cma_convergence,
        ref_power_cma,
        step_size_cma,
        step_size_rde,
        up_sampling_factor,
        bits_per_symbol,
    )
    plot_scatter(equalization_matrix_x, pt_size=1)

    # 此处均衡器内部存在一个下采样，因此均衡器出来后信号回到一个符号一个样本的采样率，也就是现在的采样率等于符号率

    return [equalization_matrix_x, equalization_matrix_y]
