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


def transmitter(
    num_symbols: int = 2**16,
    bits_per_symbol: int = 4,
    total_baud: int = 10e9,
    up_sampling_factor: int = 2,
    RRC_ROLL_OFF: float = 0.02,
    sampling_rate_awg: int = 96e9,
    dac_resolution_bits: int = 8,
    linewidth_tx: int = 150e3,
    osnr_db: int = 30,
) -> tuple[list, list]:
    """transmitter function

    Args:
        num_symbols (int): 符号数目
        bits_per_symbol (int): 2 for QPSK; 4 for 16QAM; 5 for 32QAM; 6 for 64QAM  设置调制格式
        total_baud (float): 信号波特率，符号率
        up_sampling_factor (int): 上采样倍数
        RRC_ROLL_OFF (float): RRC脉冲整形滚降系数
        sampling_rate_awg (int): DAC采样率
        dac_resolution_bits (int): DAC的bit位数
        linewidth_tx (int): 激光器线宽
        osnr_db (int): 设置系统OSNR，也就是光信号功率与噪声功率的比值，此处单位为dB

    Returns:
        signals: 发射信号
        prev_symbols: 原始信号
    """

    # 首先产生发射端X/Y双偏振信号
    bits = gen_bits(num_symbols * bits_per_symbol)  # 生成两列随机二进制序列

    # QAM调制器
    symbols = qam_modulate(bits, bits_per_symbol)

    # 此处先存储发射端原始发送信号，作为最后比较BER
    prev_symbols = symbols

    sampling_rate = up_sampling_factor * total_baud  # 信号采样率

    RRC_ROLL_OFF = 0.02  # RRC脉冲整形滚降系数
    shaper = PulseShaper(
        up_sampling_factor=up_sampling_factor,
        len_filter=128 * up_sampling_factor,
        alpha=RRC_ROLL_OFF,
        ts=1 / total_baud,
        fs=sampling_rate,
    )

    signals = shaper.tx_shape(symbols)

    """ 加入AWG中DAC的量化噪声 """
    # sampling_rate_awg = 96e9  # DAC采样率
    # dac_resolution_bits = 8  # DAC的bit位数

    signals = dac_noise(signals, sampling_rate_awg,
                        sampling_rate, dac_resolution_bits)

    """ 加入发射端激光器产生的相位噪声 """
    # linewidth_tx = 150e3  # 激光器线宽
    signals = phase_noise(
        signals, sampling_rate / total_baud, linewidth_tx, total_baud)

    """ 根据设置的OSNR来加入高斯白噪声 """
    # osnr_db = 30  # 设置系统OSNR，也就是光信号功率与噪声功率的比值，此处单位为dB
    signals = gaussian_noise(signals, osnr_db, sampling_rate)

    return signals, prev_symbols
