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


def linewidth_induced_noise(num_symbols, over_sampling_rate, linewidth, symbol_rate):
    """

    Args:
        num_symbols: 输入信号长度
        over_sampling_rate: 采样率大小
        linewidth: 线宽
        symbol_rate: 符号速率

    Returns:
        phase_noise
    """
    # std_PhaseNoise= √(2πυ/采样率)   方差公式
    std_phase_noise = np.sqrt(2 * np.pi * linewidth / (symbol_rate * over_sampling_rate))

    # 生成的矩阵为均值为0，方差为std_phase_noise的数据
    rng = np.random.default_rng()  # Numpy Random Number Generator
    phase_noise_temp = std_phase_noise * rng.standard_normal((int(num_symbols * over_sampling_rate), 1))

    phase_noise = np.zeros((num_symbols, 1))  # 定一个num_symbols行，1列的0矩阵

    phase_noise[0] = phase_noise_temp[0]

    for idx in range(1, num_symbols):
        phase_noise[idx] = phase_noise[idx - 1] + phase_noise_temp[idx]

    #   对任意的t>s>=0，增量W(t)-W(s)~N(0,σ^2(t-s）），且s>0维纳过程的定义
    #  w =a+ b.*randn(m,n);
    # 其中：a为均值；
    #            b为标准差；需要开根号
    #            m为需要产生几行；
    #           n为需要产生几列.

    return np.reshape(phase_noise, (-1, 1))
