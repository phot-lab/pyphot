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


def gen_bits(length):
    """
    Generate random 0 1 bits sequence

    Args:
        length: length of generated sequence
        seed: generator seed

    Returns:

    """
    data_x = np.random.randint(0, 2, (length, 1))  # 采用randint函数随机产生0 1码元序列x
    data_y = np.random.randint(0, 2, (length, 1))  # 采用randint函数随机产生0 1码元序列x
    return data_x, data_y
