import numpy as np


def load_awgn(length_vector):
    """
    Generate AWGN for I and Q of two polarizations
    Args:
        length_vector: 输入的信号长度

    Returns:
        noise_x: 生成的X路信号的噪声
        noise_y: 生成的Y路信号的噪声
        noise_power: 噪声功率
    """
    seed = 11
    std_noise = 1  # 设置随机噪声方差为1

    # 生成的是高斯白噪声，分别对实部虚部进行高斯白噪声的生成

    rng = np.random.default_rng(seed + 41)  # 设置随机数种子
    noise_i_x = std_noise / 2 * rng.standard_normal((length_vector, 1))
    rng = np.random.default_rng(seed + 161)
    noise_q_x = std_noise / 2 * rng.standard_normal((length_vector, 1))
    rng = np.random.default_rng(seed + 217)
    noise_i_y = std_noise / 2 * rng.standard_normal((length_vector, 1))
    rng = np.random.default_rng(seed + 311)
    noise_q_y = std_noise / 2 * rng.standard_normal((length_vector, 1))

    # 合成噪声信号（实部+虚部）
    noise_x = noise_i_x + 1j * noise_q_x
    noise_y = noise_i_y + 1j * noise_q_y

    # noise_power：噪声功率计算噪声功率
    noise_power = np.mean(np.square(np.abs(noise_x)) + np.square(np.abs(noise_y)))

    return noise_x, noise_y, noise_power
