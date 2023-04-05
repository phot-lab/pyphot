from ..optical import fiber_numpy, fiber_cupy, fiber_torch
from ..utils import settings
import numpy as np
from typing import Tuple


def fiber(
    signals: list,
    sampling_rate: float,
    num_spans: int,
    beta2: float,
    delta_z: float,
    gamma: float,
    alpha: float,
    span_length: int,
) -> Tuple[list, list]:

    """fiber function

    Args:
        signals (list): 输入信号
        sampling_rate (float): 信号采样率
        num_spans (int): span 数量
        beta2 (float): 仿真参数
        delta_z (float): 单步步长 (km)
        gamma (float): 仿真参数
        alpha (float): 仿真参数
        span_length (int): 一个 span 的长度 (km)

    Raises:
        RuntimeError: Unknown backend type

    Returns:
        list: 输出信号
    """

    if settings._backend == "numpy":
        return fiber_numpy(signals, sampling_rate, num_spans, beta2, delta_z, gamma, alpha, span_length)
    elif settings._backend == "cupy":
        return fiber_cupy(signals, sampling_rate, num_spans, beta2, delta_z, gamma, alpha, span_length)
    elif settings._backend == "torch":
        return fiber_torch(signals, sampling_rate, num_spans, beta2, delta_z, gamma, alpha, span_length)
    else:
        raise RuntimeError("Unknown backend type")
