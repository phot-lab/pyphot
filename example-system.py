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

    # 发射端
    signals, prev_symbols = phot.transmitter()

    """ Optical Fiber Channel """

    # 实际情况：1000公里 10米一步
    num_spans = 5  # 多少个 span (每个span经过一次放大器)
    span_length = 75  # 一个 span 的长度 (km)
    delta_z = 1  # 单步步长 (km)
    alpha = 0.2
    beta2 = 21.6676e-24
    gamma = 1.3

    signals, signals_power = phot.fiber(
        signals, sampling_rate, num_spans, beta2, delta_z, gamma, alpha, span_length)

    # 接收端
    signals, prev_symbols = phot.receiver(signals, prev_symbols)

    # 分析器画星座图
    phot.constellation_diagram(signals)

    # 分析器画眼图
    phot.eye_diagram(signals, up_sampling_factor)

    """ 此处开始计算误码率 """

    # 返回误码率和 Q 影响因子
    ber, q_factor = phot.bits_error_count(
        signals, prev_symbols, bits_per_symbol)
