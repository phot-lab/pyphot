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
from numpy.fft import fft, fftshift
from phot.utils import plot_linechart, volts_to_decibel


def fre_offset_compensation_fft(input_complex_signal_x, input_complex_signal_y, sample_rate):
    """

    Args:
        input_complex_signal_x: input complex samples at 1sample/symbol for FFT at x polarization
        input_complex_signal_y: input complex samples at 1sample/symbol for FFT at y polarization
        sample_rate:

    Returns:

    """

    # 算法 FFT-FOE 参考文献  Selmi, Mehrez, Yves Jaouen, and Philippe Ciblat.
    # "Accurate digital frequency offset estimator for coherent PolMux QAM transmission systems."
    # 2009 35th European Conference on Optical Communication. IEEE, 2009.

    # frequency offset estimation
    fft_size = len(input_complex_signal_x)  # 调整信号行列，将信号转为1列

    # 将信号转为1列
    input_complex_signal_x = np.reshape(input_complex_signal_x, (-1, 1))
    input_complex_signal_y = np.reshape(input_complex_signal_y, (-1, 1))

    # 取信号的4次方
    complex_signal_4th_power_x = input_complex_signal_x ** 4
    complex_signal_4th_power_y = input_complex_signal_y ** 4

    # 取信号的傅里叶变换后，求功率
    power_spectrum_x = fftshift(np.square(np.abs(fft(complex_signal_4th_power_x, axis=0))))
    power_spectrum_y = fftshift(np.square(np.abs(fft(complex_signal_4th_power_y, axis=0))))

    # PowerSpectrum_x=fftshift((abs(fft(ComplexSignal_4thPower_x)).^2).'); %% 取信号的傅里叶变换后，求功率

    power_spectrum = power_spectrum_x + power_spectrum_y  # 对X/Y偏振的功率谱相加

    # step为float型时，arange函数会产生精度问题，故转而采用linspace
    # frequency_scale = np.arange(-sample_rate / 8, sample_rate / 8, sample_rate / fft_size / 4).reshape(
    #     (-1, 1))  # Hz  列出信号的频率坐标

    # Hz  列出信号的频率坐标
    frequency_scale = np.linspace(-sample_rate / 8, sample_rate / 8, fft_size + 1)
    frequency_scale = np.delete(frequency_scale, -1).reshape((-1, 1))

    plot_linechart(frequency_scale * 1e-9, volts_to_decibel(power_spectrum))  # 画图

    max_index = np.argmax(power_spectrum)  # 找到最大的功率谱对应的位置
    fre_offset = frequency_scale[max_index][0]  # 找到最大的功率谱对应的频率坐标就是频偏
    phase_fre_offset = 2 * np.pi * fre_offset / sample_rate  # rad  利用公式计算频偏带来的相位
    foc_vector = np.reshape(np.conj(np.arange(1, len(input_complex_signal_x) + 1)).T * phase_fre_offset,
                            (-1, 1))  # 对信号添加频偏带来的相位

    output_x = input_complex_signal_x * np.exp(-1j * foc_vector)
    output_y = input_complex_signal_y * np.exp(-1j * foc_vector)

    return output_x, output_y, fre_offset
