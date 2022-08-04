import numpy as np
from ..optical import linewidth_induced_noise


class PhaseNoise:
    def __init__(self, over_sampling_rate, linewidth, symbol_rate):
        self.over_sampling_rate = over_sampling_rate
        self.linewidth = linewidth
        self.symbol_rate = symbol_rate

    def _add(self, signal):
        phase_noise = linewidth_induced_noise(len(signal), self.over_sampling_rate, self.linewidth,
                                              self.symbol_rate)
        # 添加相位噪声
        return signal * np.exp(1j * phase_noise)

    def add(self, signal_x, signal_y):
        return self._add(signal_x), self._add(signal_y)
