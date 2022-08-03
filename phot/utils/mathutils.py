import numpy as np


def gen_bits(shape: np.shape, seed=None):
    """
    Generate random 0 1 bits sequence

    Args:
        shape: shape of generated sequence
        seed: generator seed

    Returns:

    """
    rng = np.random.default_rng(seed=seed)
    data_x = rng.integers(0, 2, shape)  # 采用randint函数随机产生0 1码元序列x
    data_y = rng.integers(0, 2, shape)  # 采用randint函数随机产生0 1码元序列y
    return data_x, data_y
