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
from scipy.signal import resample_poly
from ..optical import adc_resolution, gram_schmidt_orthogonalize, cdc
from phot import logger


def add_freq_offset(signals, frequency_offset, sampling_rate):
    """添加收发端激光器造成的频偏，就是发射端激光器和接收端激光器的中心频率的偏移差"""

    # 2*pi*N*V*T   通过公式计算频偏造成的相位，N表示每个符号对应的序号，[1:length(TxSignal_X)]
    phase_carrier_offset = (
        np.arange(1, len(signals[0]) + 1).T * 2 * np.pi * frequency_offset / sampling_rate
    ).reshape((-1, 1))

    return_signals = []
    for signal in signals:

        # 添加频偏，频偏也可以看作一个相位
        signal = signal * np.exp(1j * phase_carrier_offset)

        return_signals.append(signal)

    return return_signals


def add_iq_imbalance(signals):

    signal_x = signals[0]
    signal_y = signals[1]
    rx_xi_tem = np.real(signal_x)  # 取信号实部
    rx_xq_tem = np.imag(signal_x)  # 取信号虚部
    rx_yi_tem = np.real(signal_y)  # 取信号实部
    rx_yq_tem = np.imag(signal_y)  # 取信号虚部

    amplitude_imbalance = np.power(10, 3 / 20)  # 10^(3/20)为幅度失衡因子
    phase_imbalance = np.pi * 80 / 180  # 80/180为相位失衡因子

    # 对虚部信号乘一个幅度失衡因子
    rx_xq_tem = rx_xq_tem * amplitude_imbalance
    rx_yq_tem = rx_yq_tem * amplitude_imbalance

    # 对虚部信号添加一个失衡相位,并将实部虚部组合为复数信号
    signal_x = rx_xi_tem + np.exp(1j * phase_imbalance) * rx_xq_tem
    signal_y = rx_yi_tem + np.exp(1j * phase_imbalance) * rx_yq_tem
    return [signal_x, signal_y]


def adc_noise(signals, sampling_rate, adc_sample_rate, adc_resolution_bits):
    """加入ADC的量化噪声"""

    return_signals = []

    for signal in signals:
        # 重采样改变采样率为ADC采样率，模拟进入ADC
        signal = resample_poly(signal, int(adc_sample_rate), int(sampling_rate))

        # 对信号量化，添加ADC造成的量化噪声
        signal = adc_resolution(signal, adc_resolution_bits)

        # 减去信号量化后的直流
        signal = signal - np.mean(signal)

        # 将信号采样率重采样为原来的采样率
        signal = resample_poly(signal, int(sampling_rate), int(adc_sample_rate))

        return_signals.append(signal)

    return return_signals


def iq_compensation(signals, signals_power, sampling_rate, beta2, span, L):
    """IQ正交化补偿，就是将之前的I/Q失衡的损伤补偿回来"""

    signal_x = signals[0]
    signal_y = signals[1]

    # 利用GSOP算法对I/Q失衡进行补偿，具体算法原理可看函数内部给的参考文献或论文
    rx_xi_tem, rx_xq_tem = gram_schmidt_orthogonalize(np.real(signal_x), np.imag(signal_x))
    rx_yi_tem, rx_yq_tem = gram_schmidt_orthogonalize(np.real(signal_y), np.imag(signal_y))

    # 对补偿后的实部和虚部信号进行重组
    re_x = rx_xi_tem + 1j * rx_xq_tem
    re_y = rx_yi_tem + 1j * rx_yq_tem

    re_x, re_y = cdc(re_x, re_y, signals_power[0], signals_power[1], sampling_rate, beta2, span, L)
    return [re_x, re_y]
