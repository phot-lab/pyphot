from ..optical import fiber_numpy, fiber_cupy, fiber_torch
from ..utils import settings


def fiber(signals, sampling_rate: float, span, num_steps, beta2, delta_z, gamma, alpha, L):
    if settings._backend == "numpy":
        return fiber_numpy(signals, sampling_rate, span, num_steps, beta2, delta_z, gamma, alpha, L)
    elif settings._backend == "cupy":
        return fiber_cupy(signals, sampling_rate, span, num_steps, beta2, delta_z, gamma, alpha, L)
    elif settings._backend == "torch":
        return fiber_torch(signals, sampling_rate, span, num_steps, beta2, delta_z, gamma, alpha, L)
    else:
        raise RuntimeError("Unknown backend type")
