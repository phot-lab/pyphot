from .signal import awgn
import numpy as np


def elec_amp(signal, gain_ea, power, spec_density):
    """
    Electrc Amplifier
    :param signal: The input signal
    :param gain_ea: Gain EA
    :param power: Power of the input signal
    :param spec_density: One sided spectral density
    :return: Amplified signal
    """
    power_after_ea = power * np.power(10, gain_ea / 10)
    signal = signal * np.sqrt(power_after_ea)
    snr_gs = 10 * np.log10(power / spec_density)  # SNR GS
    signal = awgn(signal, snr_gs)
    return signal
