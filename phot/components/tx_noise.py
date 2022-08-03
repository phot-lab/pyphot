from .dac_noise import DacNoise
from .phase_noise import PhaseNoise
from .gaussian_noise import GaussianNoise


class TxNoise:
    def __init__(self, sampling_rate_awg, dac_resolution_bits, sampling_rate, linewidth_tx, total_baud):
        self.sampling_rate_awg = sampling_rate_awg
        self.dac_resolution_bits = dac_resolution_bits
        self.sampling_rate = sampling_rate
        self.linewidth_tx = linewidth_tx
        self.total_baud = total_baud

    def add(self, signal_x, signal_y):
        """ 发射端噪声 """

        """ 加入AWG中DAC的量化噪声 """
        noise = DacNoise(self.sampling_rate_awg, self.sampling_rate, self.dac_resolution_bits)

        signal_x = noise.add(signal_x)
        signal_y = noise.add(signal_y)

        """ 加入发射端激光器产生的相位噪声 """

        noise = PhaseNoise(len(signal_x), self.sampling_rate / self.total_baud, self.linewidth_tx, self.total_baud)

        signal_x = noise.add(signal_x)
        signal_y = noise.add(signal_y)

        """ 根据设置的OSNR来加入高斯白噪声 """

        osnr_db = 25  # 设置系统OSNR，也就是光信号功率与噪声功率的比值，此处单位为dB
        noise = GaussianNoise(osnr_db, self.sampling_rate)

        signal_x, signal_y = noise.add(signal_x, signal_y)

        """ 添加接收端激光器产生的相位噪声 """

        noise = PhaseNoise(len(signal_x), self.sampling_rate / self.total_baud, self.linewidth_tx, self.total_baud)

        signal_x = noise.add(signal_x)
        signal_y = noise.add(signal_y)

        return signal_x, signal_y
