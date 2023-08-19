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

import numpy as np


def signal_power(signals, power_dbm_set=-6):
    data_x = signals[0]
    data_y = signals[1]

    data_x_normal = data_x / np.sqrt(np.mean(abs(data_x) ** 2))
    data_y_normal = data_y / np.sqrt(np.mean(abs(data_y) ** 2))

    # 信号功率
    # power_dbm_set = -6
    power_mw = 10 ** (power_dbm_set / 10)
    power_w = power_mw / 1000

    power_origin_x = np.mean(np.abs(data_x_normal) ** 2)
    power_origin_y = np.mean(np.abs(data_y_normal) ** 2)

    data_x = data_x_normal / np.sqrt(power_origin_x / power_w)
    data_y = data_y_normal / np.sqrt(power_origin_y / power_w)

    power_temp_x = np.mean(np.abs(data_x) ** 2)
    power_temp_y = np.mean(np.abs(data_y) ** 2)

    return [power_temp_x, power_temp_y]
