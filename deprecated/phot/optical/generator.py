import numpy as np
from deprecated.phot.optical import format
from deprecated.phot.values import globals


def rand(size, seed=None):
    if seed is None:
        return np.random.random(size)
    else:
        rng = np.random.RandomState(seed)
        return rng.random(size)


def gen_seq(seq_type: str, mod_format="dpsk", seed=None) -> np.ndarray:
    """
    Generate random number sequence
    :param num_sym: number of symbols
    :param seq_type: random sequence type
    :param mod_format: modulation format
    :param seed: the seed of generator, will be random if not fixed
    :return: random number sequence
    """

    if seq_type == "rand":  # Random Uniformly-distributed sequence
        format_info = format.get_format_info(mod_format)
        digit = format_info.digit
        seq = np.floor(rand((globals._num_sym, 1), seed) * digit)
    else:
        raise RuntimeError('Unknown random sequence type')
    return seq
