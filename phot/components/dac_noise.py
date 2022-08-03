from scipy.signal import resample_poly
from ..optical import dac_resolution
import numpy as np


class DacNoise:
    def __init__(self, sampling_rate_awg, sampling_rate, dac_resolution_bits):
        self.sampling_rate_awg = sampling_rate_awg
        self.sampling_rate = sampling_rate
        self.dac_resolution_bits = dac_resolution_bits

    def add(self, signal):
        # 重采样信号采样率为DAC的采样率，模拟进入DAC
        signal = resample_poly(signal, int(self.sampling_rate_awg), int(self.sampling_rate))

        # 对信号进行量化，模拟加入量化噪声
        signal = dac_resolution(signal, self.dac_resolution_bits)

        # 减去信号量化后的直流，也就是均值
        signal = signal - np.mean(signal)

        # 重采样信号采样率为原来的采样率，模拟出DAC后采样率，resample_poly 为SciPy库的函数
        signal = resample_poly(signal, int(self.sampling_rate), int(self.sampling_rate_awg)).reshape((-1, 1))

        return signal
