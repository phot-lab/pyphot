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


def phase_noise(signals, over_sampling_rate, linewidth, symbol_rate):
    phase_noise = linewidth_induced_noise(len(signals[0]), over_sampling_rate, linewidth, symbol_rate)

    # 添加相位噪声
    return_signals = []
    for signal in signals:
        signal = signal * np.exp(1j * phase_noise)
        return_signals.append(signal)
    return return_signals
