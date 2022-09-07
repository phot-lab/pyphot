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
from phot.utils import quantize


def dac_resolution(signal, resolution_bit):
    """
    DAC Resolution
    Args:
        signal: 输入信号
        resolution_bit: 量化等级

    Returns:
        out: 输出信号
    """
    signal_i = np.real(signal)
    signal_q = np.imag(signal)

    # 把信号分为了实部虚部，分别计算实部虚部最大的长度（定义量化区间）
    max_i = np.max(np.abs(signal_i))
    max_q = np.max(np.abs(signal_q))
    a = min(max_i, max_q)  # 取两个值的最小值为A

    # 每个区间映射的离散数值计算量化值，A的值为1.44 区间[-1.44,1.44]一共分为2^resolution_bit 份
    codebook = np.linspace(-a, a, 2 ** resolution_bit)

    # 分割向量，计算对量化值的分割等级   取中间值 中间值的数量比总体少一
    partition = codebook - (codebook[1] - codebook[0]) / 2

    partition = np.delete(partition, 0)  # 256变为255

    # 量化过后的输出，实部虚部均进行量化
    output_i, samples_quant_i = quantize(signal_i.ravel(), partition, codebook)
    output_q, samples_quant_q = quantize(signal_q.ravel(), partition, codebook)

    out = output_i + 1j * output_q  # 输出信号=实部+j虚部

    # [index等级索引,quants按照索引取到的量化值codebook] =quantiz(sig,partit ion 分割等级 ,codebook量化值)
    # index sig 中数值落入到partition的某个区间对应的编号，区间编号从0开始
    # quants 是 sig 中数值对应的量化后的离散数值，所有离散取值的集合是 codebook 。
    # 因为 index 是从0开始编号，所以，quants 与 index 的具体对应关系是：quants(k) = codebook(index(k) + 1)

    return out


def adc_resolution(signal, resolution_bit) -> np.ndarray:
    """
    ADC Resolution
    Args:
        signal: 输入信号
        resolution_bit: 量化等级

    Returns:
        out: 输出信号
    """
    signal_i = np.real(signal)
    signal_q = np.imag(signal)

    # 把信号分为了实部虚部，分别计算实部虚部最大的长度（定义量化区间）
    max_i = np.max(np.abs(signal_i))
    max_q = np.max(np.abs(signal_q))
    a = min(max_i, max_q)  # 取两个值的最小值为A

    # 每个区间映射的离散数值计算量化值，A的值为1.44 区间[-1.44,1.44]一共分为2^resolution_bit 份
    codebook = np.linspace(-a, a, 2 ** resolution_bit)

    # 分割向量，计算对量化值的分割等级   取中间值 中间值的数量比总体少一
    partition = codebook - (codebook[1] - codebook[0]) / 2

    partition = np.delete(partition, 0)  # 256变为255

    # 量化过后的输出，实部虚部均进行量化
    output_i, samples_quant_i = quantize(signal_i.ravel(), partition, codebook)
    output_q, samples_quant_q = quantize(signal_q.ravel(), partition, codebook)

    out = output_i + 1j * output_q  # 输出信号=实部+j虚部

    # [index等级索引,quants按照索引取到的量化值codebook] =quantiz(sig,partit ion 分割等级 ,codebook量化值)
    # index sig 中数值落入到partition的某个区间对应的编号，区间编号从0开始
    # quants 是 sig 中数值对应的量化后的离散数值，所有离散取值的集合是 codebook 。
    # 因为 index 是从0开始编号，所以，quants 与 index 的具体对应关系是：quants(k) = codebook(index(k) + 1)

    return np.reshape(out, (-1, 1))
