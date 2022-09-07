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
from scipy.signal import lfilter


def fine_synchronize(tr_data_x, re_training_x):
    # Method 1: the same to zhuge's
    # 采用自相关的同步算法，找到自相关性最大的位置

    # 错误信号= 取绝对值(filter滤波器(左右反转(取共轭(要同步的信号)),1,对应的正确的信号));
    matrix_error = np.abs(lfilter(np.fliplr(np.conj(re_training_x)).ravel(), 1, tr_data_x.ravel()))

    n = np.abs(np.sum(re_training_x * np.conj(re_training_x)))  # N的大小=取绝对值(求和(要同步的信号.*取共轭(要同步的信号)));
    matrix_error = matrix_error / n
    start_index = np.argmax(matrix_error)  # 找的最大的错误信号值，并确定最大错误值的位置序号
    start_index = start_index - len(re_training_x[0]) + 1  # 输出信号开始的同步序列号= 最大错误值的序号-要同步的信号的长度

    # % figure; plot(matric_error);

    return start_index
