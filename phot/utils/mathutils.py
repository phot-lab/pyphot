import numpy as np


def gen_bits(shape: np.shape, seed=None):
    """
    Generate random 0 1 bits sequence

    Args:
        shape: shape of generated sequence
        seed: generator seed

    Returns:

    """
    rng = np.random.RandomState(seed)
    arr = rng.rand(shape[0], shape[1])
    arr = np.round(arr).astype(dtype=int)
    return arr
