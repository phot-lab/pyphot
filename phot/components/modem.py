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

from commpy.modulation import QAMModem


def qam_modulate(data_x, data_y, bits_per_symbol):
    """QAM modulate

    Args:
        data_x (np.ndarray): signal x
        data_y (np.ndarray): signal y
        bits_per_symbol (int/float): Bits per symbol
    """
    modem = QAMModem(2**bits_per_symbol)
    symbols_x = modem.modulate(data_x).reshape((-1, 1))  # X偏振信号
    symbols_y = modem.modulate(data_y).reshape((-1, 1))  # Y偏振信号
    return symbols_x, symbols_y
