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

from ..optical import bps_hybrid_qam
from ..utils import plot_scatter
import numpy as np


class BPS:
    def __init__(self, num_test_angle, block_size, bits_per_symbol):
        self.num_test_angle = num_test_angle
        self.block_size = block_size
        self.bits_per_symbol = bits_per_symbol

    def restore(self, signal_x, signal_y):
        # BPS算法
        equalization_matrix_x, phase_x = bps_hybrid_qam(np.real(signal_x), np.imag(signal_x), self.num_test_angle,
                                                        self.block_size, self.bits_per_symbol)
        equalization_matrix_y, phase_y = bps_hybrid_qam(np.real(signal_y), np.imag(signal_y), self.num_test_angle,
                                                        self.block_size, self.bits_per_symbol)
        """ normalization 对信号进行归一化，代码过程中存在一个信号的缩放问题，此处将信号恢复回去 """
        if self.bits_per_symbol == 2:
            equalization_matrix_x = equalization_matrix_x * np.sqrt(2)  # 不同调制格式对应的归一化因子不同
            equalization_matrix_y = equalization_matrix_y * np.sqrt(2)
        elif self.bits_per_symbol == 4:
            equalization_matrix_x = equalization_matrix_x * np.sqrt(10)
            equalization_matrix_y = equalization_matrix_y * np.sqrt(10)
        elif self.bits_per_symbol == 5:
            equalization_matrix_x = equalization_matrix_x * np.sqrt(20)
            equalization_matrix_y = equalization_matrix_y * np.sqrt(20)

        plot_scatter(equalization_matrix_x)

        return equalization_matrix_x, equalization_matrix_y
