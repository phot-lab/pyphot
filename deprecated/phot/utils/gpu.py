from deprecated.phot.utils import logger
import numpy as np


def to_gpu(array):
    try:
        import cupy as cp
        if isinstance(array, cp._core.core.ndarray):
            return array
        return cp.asarray(array)
    except ModuleNotFoundError:
        logger.warning("CUDA is not available, use CPU computation instead")
        return array


def to_cpu(array):
    if isinstance(array, np.ndarray):
        return array
    try:
        import cupy as cp
        return cp.asnumpy(array)
    except ModuleNotFoundError:
        logger.warning("CUDA is not available, use CPU computation instead")
        return array
