"""
Copyright (c) 2022 Beijing Jiaotong University
PhotLab is licensed under [Open Source License].
You can use this software according to the terms and conditions of the [Open Source License].
You may obtain a copy of [Open Source License] at: [https://open.source.license/]

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.

See the [Open Source License] for more details.

Author: Zhenjie Wei
Created: 2023/8/18
Supported by: National Key Research and Development Program of China
"""

from ..optical import *
from ..utils import *
from . import *


def receiver(
    signals,
    prev_symbols,
    bits_per_symbol: int = 4,
    total_baud: int = 10e9,
    up_sampling_factor: int = 2,
    RRC_ROLL_OFF: float = 0.02,
    num_spans: int = 5,
    beta2: float = 21.6676e-24,
    span_length: int = 75,
    linewidth_rx: int = 150e3,
    frequency_offset: float = 2e9,
    adc_sample_rate: float = 160e9,
    adc_resolution_bits: int = 8,
    num_tap: int = 25,
    ref_power_cma: int = 2,
    cma_convergence: int = 30000,
    step_size_cma: float = 1e-9,
    step_size_rde: float = 1e-9,
    num_test_angle: int = 64,
    block_size: int = 100
) -> list:
    """receiver function

    Args:
        bits_per_symbol (int): 2 for QPSK; 4 for 16QAM; 5 for 32QAM; 6 for 64QAM  设置调制格式
        total_baud (int): 信号波特率，符号率
        up_sampling_factor (int): 上采样倍数
        RRC_ROLL_OFF (float): RRC脉冲整形滚降系数
        signal_power (int): 信号功率
        num_spans (int): span 数量 
        beta2 (float): 仿真参数
        span_length (int): 一个 span 的长度 (km)
        linewidth_rx (int): 激光器线宽
        frequency_offset (float): 设置频偏，一般激光器的频偏范围为 -3G~3G Hz
        adc_sample_rate (float): ADC采样率
        adc_resolution_bits (int): ADC的bit位数
        num_tap (int): 均衡器抽头数目，此处均衡器内部是采用FIR滤波器，具体可查阅百度或者论文
        ref_power_cma (int): 设置CMA算法的模
        cma_convergence (int): CMA预均衡收敛的信号长度
        step_size_cma (float): CMA的更新步长，梯度下降法的步长
        step_size_rde (float): RDE的更新步长，梯度下降法的步长，%% CMA和RDE主要就是损失函数不同
        num_test_angle (int): BPS算法的测试角数目，具体算法原理可以参考函数内部给的参考文献
        block_size (int): BPS算法的块长设置

    Returns:
        signals: 接收信号
    """

    sampling_rate = up_sampling_factor * total_baud  # 信号采样率
    signals_power = signal_power(signals)

    """ 添加接收端激光器产生的相位噪声 """
    # linewidth_rx = 150e3  # 激光器线宽
    signals = phase_noise(
        signals, sampling_rate / total_baud, linewidth_rx, total_baud)

    """ 添加收发端激光器造成的频偏，就是发射端激光器和接收端激光器的中心频率的偏移差 """
    # frequency_offset = 2e9  # 设置频偏，一般激光器的频偏范围为 -3G~3G Hz
    signals = add_freq_offset(signals, frequency_offset, sampling_rate)

    """ 模拟接收机造成的I/Q失衡，主要考虑幅度失衡和相位失衡，这里将两者都加在虚部上 """
    signals = add_iq_imbalance(signals)

    """ 加入ADC的量化噪声 """
    # adc_sample_rate = 160e9  # ADC采样率
    # adc_resolution_bits = 8  # ADC的bit位数

    signals = adc_noise(signals, sampling_rate,
                        adc_sample_rate, adc_resolution_bits)

    """ IQ正交化补偿，就是将之前的I/Q失衡的损伤补偿回来 """
    signals = iq_compensation(
        signals, signals_power, sampling_rate, beta2, num_spans, span_length)

    """ 
    粗糙的频偏估计和补偿，先进行一个频偏的补偿，
    因为后面有一个帧同步，而帧同步之前需要先对频偏进行补偿，否则帧同步不正确
    """
    signals = freq_offset_compensation(signals, sampling_rate)

    """ 接收端相应的RRC脉冲整形，具体的参数代码与发射端的RRC滤波器是一致的 """
    RRC_ROLL_OFF = 0.02  # RRC脉冲整形滚降系数
    shaper = PulseShaper(
        up_sampling_factor=up_sampling_factor,
        len_filter=128 * up_sampling_factor,
        alpha=RRC_ROLL_OFF,
        ts=1 / total_baud,
        fs=sampling_rate,
    )
    signals = shaper.rx_shape(signals)

    """ 帧同步，寻找与发射端原始信号头部对应的符号 """
    signals, prev_symbols = sync_frame(
        signals, prev_symbols, up_sampling_factor)

    """ 
    自适应均衡，此处采用恒模算法（CMA）对收敛系数进行预收敛，
    再拿收敛后的滤波器系数对正式的信号使用半径定向算法（RDE）进行均衡收敛，
    总的思想采用梯度下降法 
    """

    # num_tap = 25  # 均衡器抽头数目，此处均衡器内部是采用FIR滤波器，具体可查阅百度或者论文
    # ref_power_cma = 2  # 设置CMA算法的模
    # cma_convergence = 30000  # CMA预均衡收敛的信号长度
    # step_size_cma = 1e-9  # CMA的更新步长，梯度下降法的步长
    # step_size_rde = 1e-9  # RDE的更新步长，梯度下降法的步长，%% CMA和RDE主要就是损失函数不同

    signals = adaptive_equalize(
        signals,
        num_tap,
        cma_convergence,
        ref_power_cma,
        step_size_cma,
        step_size_rde,
        up_sampling_factor,
        bits_per_symbol,
        total_baud,
    )

    """ 
    均衡后进行精确的频偏估计和补偿 采用FFT-FOE算法，
    与前面的粗估计一样，防止前面粗估计没补偿完全，此处做一个补充 
    """
    signals = freq_offset_compensation(signals, total_baud)

    """ 相位恢复  采用盲相位搜索算法（BPS）进行相位估计和补偿 """

    # num_test_angle = 64  # BPS算法的测试角数目，具体算法原理可以参考函数内部给的参考文献
    # block_size = 100  # BPS算法的块长设置

    signals = bps_restore(
        signals, num_test_angle, block_size, bits_per_symbol)

    return signals, prev_symbols
