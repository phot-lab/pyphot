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

from .logging import logger
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


def is_gpu(array):
    try:
        import cupy as cp
        if isinstance(array, cp._core.core.ndarray):
            return True
        else:
            return False
    except ModuleNotFoundError:
        logger.warning("CUDA is not available, use CPU computation instead")
        return array
