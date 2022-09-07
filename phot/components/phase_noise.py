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
from ..optical import linewidth_induced_noise


class PhaseNoise:
    def __init__(self, over_sampling_rate, linewidth, symbol_rate):
        self.over_sampling_rate = over_sampling_rate
        self.linewidth = linewidth
        self.symbol_rate = symbol_rate

    def _add(self, signal):
        phase_noise = linewidth_induced_noise(len(signal), self.over_sampling_rate, self.linewidth,
                                              self.symbol_rate)
        # 添加相位噪声
        return signal * np.exp(1j * phase_noise)

    def add(self, signal_x, signal_y):
        return self._add(signal_x), self._add(signal_y)
