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

from scipy.signal import resample_poly
from ..optical import dac_resolution
import numpy as np


class DacNoise:
    def __init__(self, sampling_rate_awg, sampling_rate, dac_resolution_bits):
        self.sampling_rate_awg = sampling_rate_awg
        self.sampling_rate = sampling_rate
        self.dac_resolution_bits = dac_resolution_bits

    def _add(self, signal):
        # 重采样信号采样率为DAC的采样率，模拟进入DAC
        signal = resample_poly(signal, int(self.sampling_rate_awg), int(self.sampling_rate))

        # 对信号进行量化，模拟加入量化噪声
        signal = dac_resolution(signal, self.dac_resolution_bits)

        # 减去信号量化后的直流，也就是均值
        signal = signal - np.mean(signal)

        # 重采样信号采样率为原来的采样率，模拟出DAC后采样率，resample_poly 为SciPy库的函数
        signal = resample_poly(signal, int(self.sampling_rate), int(self.sampling_rate_awg)).reshape((-1, 1))

        return signal

    def add(self, signal_x, signal_y):
        return self._add(signal_x), self._add(signal_y)
