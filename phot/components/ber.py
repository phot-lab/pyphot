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
from ..optical import fine_synchronize, ber_count, ber_estimate
from commpy.modulation import QAMModem
from phot import logger


def bits_error_count(signal_x, signal_y, prev_signal_x, prev_signal_y, bits_per_symbol):
    """计算误码率"""

    """ 再进行一个帧同步，因为经过均衡器会存在符号的一些舍弃，因此在计算误码率（BER）之前需要再一次帧同步 """

    # 对发射端信号跟均衡后信号进行同步
    start_index_x_1 = fine_synchronize(prev_signal_x[:, 0].T, signal_x[0:10000, 0].reshape((1, -1)))

    # 对发射端信号利用自带函数进行移动
    prev_signal_x = np.roll(prev_signal_x, -start_index_x_1)
    prev_signal_y = np.roll(prev_signal_y, -start_index_x_1)

    # 变为16的倍数
    signal_x = signal_x[:-1]
    signal_y = signal_y[:-1]

    # 使得均衡后信号与发射端信号一样长度
    prev_signal_x = prev_signal_x[0 : len(signal_x), :]
    prev_signal_y = prev_signal_y[0 : len(signal_y), :]

    """ BER COUNT  对信号进行误码率计算，将接收信号与发射信号转化为格雷编码，比较各个码元的正确率 """

    ber, q_db = ber_estimate(prev_signal_y, signal_y, bits_per_symbol)
    logger.info("Estimated overall bits error is {:.5f}".format(ber))
    logger.info("Estimated Q factor is {:.5f}".format(q_db))

    # 下面是遗弃版本
    # modem = QAMModem(2**bits_per_symbol)

    # tx_bits = modem.demodulate(
    #     np.concatenate((np.reshape(prev_signal_x, (-1, 1)), np.reshape(prev_signal_y, (-1, 1))), axis=0).ravel(),
    #     demod_type="hard",
    # )
    # rx_bits = modem.demodulate(
    #     np.concatenate((np.reshape(signal_x, (-1, 1)), np.reshape(signal_y, (-1, 1))), axis=0).ravel(),
    #     demod_type="hard",
    # )
    # ber, q_db = ber_count(rx_bits, tx_bits)  # 比较码元的正确率
    # logger.info("Calculated overall bits error is {:.5f}".format(ber))
