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

from ..optical import fine_synchronize
import numpy as np
from phot import logger


def sync_frame(signal_x, signal_y, prev_signal_x, prev_signal_y, up_sampling_factor):
    """帧同步，寻找与发射端原始信号头部对应的符号"""

    # 对信号进行帧同步，找出接收信号与发射信号对准的开头
    start_index_x_1 = fine_synchronize(
        signal_x[0 : 10000 * up_sampling_factor : up_sampling_factor].T, prev_signal_x[0:4000].T
    )
    start_index_y_1 = fine_synchronize(
        signal_y[0 : 10000 * up_sampling_factor : up_sampling_factor].T, prev_signal_y[0:4000].T
    )

    logger.info("两个偏振第一次对准的帧头")
    logger.info("Start_Index_X_1: {} Start_Index_Y_1: {}".format(start_index_x_1, start_index_y_1))

    # 将帧头位置前的信号去除，以便发射端信号与接收端信号的头部对准
    signal_x = np.delete(signal_x, np.arange(0, start_index_x_1 * up_sampling_factor))
    signal_y = np.delete(signal_y, np.arange(0, start_index_y_1 * up_sampling_factor))

    # 去掉接收信号的尾部部分信号
    signal_x = signal_x[:-1000]
    signal_y = signal_y[:-1000]

    # 调整发射端信号与接收端信号的长度
    prev_signal_x = prev_signal_x[0 : int(np.floor(len(signal_x) / up_sampling_factor))]
    prev_signal_y = prev_signal_y[0 : int(np.floor(len(signal_y) / up_sampling_factor))]

    return signal_x, signal_y, prev_signal_x, prev_signal_y
