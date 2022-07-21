import numpy as np


def gram_schmidt_orthogonalize(input_i, input_q):
    # Ensure length of I,Q data is the same!
    length_min = min(len(input_i), len(input_q))  # 调整信号的长度，保证I,Q路的信号长度一致

    input_i = np.reshape(input_i[0:length_min], (length_min, 1))
    input_q = np.reshape(input_q[0:length_min], (length_min, 1))

    # I acted as the reference orientation

    average_power_i = np.mean(input_i ** 2)  # 对I路信号的平方求平均值
    correction_coefficient = np.mean(input_i * input_q)  # 对I,Q两路信号进行叉乘处理，并取平均值
    reference = input_q - (correction_coefficient / average_power_i) * input_i
    average_power_q = np.mean(reference ** 2)
    output_q = reference / np.sqrt(average_power_q)
    output_i = input_i / np.sqrt(average_power_i)

    # 算法GSOP参考文献  Fatadin, Irshaad, Seb J. Savory, and David Ives.
    # "Compensation of quadrature imbalance in an optical QPSK coherent receiver."
    # IEEE Photonics Technology Letters 20.20 (2008): 1733-1735.

    return output_i, output_q
