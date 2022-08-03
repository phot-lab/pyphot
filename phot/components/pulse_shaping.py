from ..utils import upsample
from commpy.filters import rrcosfilter
import numpy as np
from scipy.signal import lfilter


class PulseShaper:
    def __init__(self, up_sampling_factor, len_filter, alpha, ts, fs):
        self.up_sampling_factor = up_sampling_factor
        self.len_filter = len_filter
        self.alpha = alpha
        self.ts = ts
        self.fs = fs

        # 产生RRC滤波器
        time_idx, rrc_filter = rrcosfilter(N=128 * self.up_sampling_factor, alpha=self.alpha, Ts=self.ts,
                                           Fs=self.fs)  # up_sampling_factor*128为滤波器的长度
        self.rrc_filter = (rrc_filter * np.sqrt(2))

    def tx_shape(self, signal_x, signal_y):
        # 先进行插0上采样，上采样倍数为 up_sampling_factor
        up_sampled_signal_x = upsample(signal_x, self.up_sampling_factor)
        up_sampled_signal_y = upsample(signal_y, self.up_sampling_factor)

        # 对信号使用RRC滤波器脉冲整形
        rrc_signal_x = lfilter(self.rrc_filter, 1, up_sampled_signal_x, axis=0)
        rrc_signal_y = lfilter(self.rrc_filter, 1, up_sampled_signal_y, axis=0)
        return rrc_signal_x, rrc_signal_y

    def rx_shape(self, signal_x, signal_y):
        signal_x = lfilter(self.rrc_filter, 1, signal_x, axis=0)
        signal_y = lfilter(self.rrc_filter, 1, signal_y, axis=0)
        return signal_x, signal_y
