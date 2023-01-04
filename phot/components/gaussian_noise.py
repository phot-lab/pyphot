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

from ..optical import load_awgn
import numpy as np


def gaussian_noise(signal_x, signal_y, osnr, sampling_rate):
    # 生成均值为0，方差为1的随机噪声,此处直接产生两个偏振的噪声
    noise_x, noise_y, noise_power = load_awgn(len(signal_x))

    # 计算当前信号功率
    original_avg_power = np.mean(np.square(np.abs(signal_x)) + np.square(np.abs(signal_x)))

    osnr = np.power(10, osnr / 10)  # 将OSNR的单位由dB转为常量单位

    # 先计算OSNR对应的SNR，再通过SNR计算所需要达到的目标的信号功率，12.5e9为信号的中心频率，此处是根据公式计算，可参考通信原理或者百度或者论文
    target_avg_power = noise_power * osnr * 12.5e9 / sampling_rate

    # 改变当前信号功率为目标功率
    signal_x = np.sqrt(target_avg_power / original_avg_power) * signal_x
    signal_y = np.sqrt(target_avg_power / original_avg_power) * signal_y

    # 对信号添加随机噪声
    signal_x = noise_x + signal_x
    signal_y = noise_y + signal_y

    return signal_x, signal_y
