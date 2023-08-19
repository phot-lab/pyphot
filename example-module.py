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
Created: 2023/8/19
Supported by: National Key Research and Development Program of China
"""

import phot

if __name__ == "__main__":
    """双偏振光收发模块 + 光纤信道"""
    """本代码为程序主函数 本代码主要适用于 QPSK，16QAM，32QAM，64QAM 调制格式的单载波相干背靠背（B2B）信号"""

    phot.config(plot=True, backend="numpy")  # 全局开启画图，backend 使用 numpy

    # 设置全局系统仿真参数
    num_symbols = 2**16  # 符号数目
    bits_per_symbol = 4  # 2 for QPSK; 4 for 16QAM; 5 for 32QAM; 6 for 64QAM  设置调制格式
    total_baud = 10e9  # 信号波特率，符号率
    up_sampling_factor = 2  # 上采样倍数
    sampling_rate = up_sampling_factor * total_baud  # 信号采样率

    """ 程序从此开始为发射端的仿真代码 """

    # 首先产生发射端X/Y双偏振信号

    bits = phot.gen_bits(num_symbols * bits_per_symbol)  # 生成两列随机二进制序列

    # QAM调制器
    symbols = phot.qam_modulate(bits, bits_per_symbol)

    # 此处先存储发射端原始发送信号，作为最后比较BER
    prev_symbols = symbols

    RRC_ROLL_OFF = 0.02  # RRC脉冲整形滚降系数
    shaper = phot.PulseShaper(
        up_sampling_factor=up_sampling_factor,
        len_filter=128 * up_sampling_factor,
        alpha=RRC_ROLL_OFF,
        ts=1 / total_baud,
        fs=sampling_rate,
    )

    signals = shaper.tx_shape(symbols)

    """ 加入AWG中DAC的量化噪声 """
    sampling_rate_awg = 96e9  # DAC采样率
    dac_resolution_bits = 8  # DAC的bit位数

    signals = phot.dac_noise(signals, sampling_rate_awg,
                             sampling_rate, dac_resolution_bits)

    """ 加入发射端激光器产生的相位噪声 """
    linewidth_tx = 150e3  # 激光器线宽
    signals = phot.phase_noise(
        signals, sampling_rate / total_baud, linewidth_tx, total_baud)

    """ 根据设置的OSNR来加入高斯白噪声 """

    osnr_db = 30  # 设置系统OSNR，也就是光信号功率与噪声功率的比值，此处单位为dB
    signals = phot.gaussian_noise(signals, osnr_db, sampling_rate)

    """ 发射端代码此处截止 """

    """ Optical Fiber Channel """

    # 实际情况：1000公里 10米一步
    num_spans = 5  # 多少个 span (每个span经过一次放大器)
    span_length = 75  # 一个 span 的长度 (km)
    delta_z = 1  # 单步步长 (km)
    alpha = 0.2
    beta2 = 21.6676e-24
    gamma = 1.3

    # signals, signals_power = phot.fiber(signals, sampling_rate, num_spans, beta2, delta_z, gamma, alpha, span_length)
    signals_power = phot.signal_power(signals)

    """ 添加接收端激光器产生的相位噪声 """
    linewidth_rx = 150e3  # 激光器线宽
    signals = phot.phase_noise(
        signals, sampling_rate / total_baud, linewidth_rx, total_baud)

    """ 添加收发端激光器造成的频偏，就是发射端激光器和接收端激光器的中心频率的偏移差 """

    frequency_offset = 2e9  # 设置频偏，一般激光器的频偏范围为 -3G~3G Hz

    signals = phot.add_freq_offset(signals, frequency_offset, sampling_rate)

    """ 模拟接收机造成的I/Q失衡，主要考虑幅度失衡和相位失衡，这里将两者都加在虚部上 """

    signals = phot.add_iq_imbalance(signals)

    """ 加入ADC的量化噪声 """
    adc_sample_rate = 160e9  # ADC采样率
    adc_resolution_bits = 8  # ADC的bit位数

    signals = phot.adc_noise(signals, sampling_rate,
                             adc_sample_rate, adc_resolution_bits)

    """ IQ正交化补偿，就是将之前的I/Q失衡的损伤补偿回来 """
    signals = phot.iq_compensation(
        signals, signals_power, sampling_rate, beta2, num_spans, span_length)

    """ 粗糙的频偏估计和补偿，先进行一个频偏的补偿，因为后面有一个帧同步，而帧同步之前需要先对频偏进行补偿，否则帧同步不正确 """
    signals = phot.freq_offset_compensation(signals, sampling_rate)

    """ 接收端相应的RRC脉冲整形，具体的参数代码与发射端的RRC滤波器是一致的 """
    signals = shaper.rx_shape(signals)

    """ 帧同步，寻找与发射端原始信号头部对应的符号 """
    signals, prev_symbols = phot.sync_frame(
        signals, prev_symbols, up_sampling_factor)

    """ 自适应均衡，此处采用恒模算法（CMA）对收敛系数进行预收敛，再拿收敛后的滤波器系数对正式的信号使用半径定向算法（RDE）进行均衡收敛，总的思想采用梯度下降法 """

    num_tap = 25  # 均衡器抽头数目，此处均衡器内部是采用FIR滤波器，具体可查阅百度或者论文，
    ref_power_cma = 2  # 设置CMA算法的模
    cma_convergence = 30000  # CMA预均衡收敛的信号长度
    step_size_cma = 1e-9  # CMA的更新步长，梯度下降法的步长
    step_size_rde = 1e-9  # RDE的更新步长，梯度下降法的步长，%% CMA和RDE主要就是损失函数不同

    signals = phot.adaptive_equalize(
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

    """ 均衡后进行精确的频偏估计和补偿 采用FFT-FOE算法，与前面的粗估计一样，防止前面粗估计没补偿完全，此处做一个补充 """
    signals = phot.freq_offset_compensation(signals, total_baud)

    """ 相位恢复  采用盲相位搜索算法（BPS）进行相位估计和补偿 """

    num_test_angle = 64  # BPS算法的测试角数目，具体算法原理可以参考函数内部给的参考文献
    block_size = 100  # BPS算法的块长设置

    signals = phot.bps_restore(
        signals, num_test_angle, block_size, bits_per_symbol)

    # 分析器画星座图
    phot.constellation_diagram(signals)

    # 分析器画眼图
    phot.eye_diagram(signals, up_sampling_factor)

    """ 此处开始计算误码率 """

    # 返回误码率和 Q 影响因子
    ber, q_factor = phot.bits_error_count(
        signals, prev_symbols, bits_per_symbol)
