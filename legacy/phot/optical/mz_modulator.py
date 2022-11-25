import math
import numpy as np
from deprecated.phot.utils import mathutils
from .lightwave import Lightwave


def mz_modulator(lightwave, sig, options=None) -> Lightwave:
    # Default values
    bias_lower = -1  # bias of lower arm
    bias_upper = -1  # bias of upper arm
    ext_ratio = math.inf  # extinction ratio
    pp = 0  # 0: push-pull. 1: push-push
    voltage_pi = math.pi / 2 * (pp + 1)  # voltage of phase shift pi
    norm_factor = 1  # normalization factor

    if np.size(lightwave.lam) > 1:  # need to know which channel to modulate
        if options is None or hasattr(options, "num_chl"):
            raise RuntimeError("Missing channel index")
        num_chl = options.num_chl
    else:
        num_chl = 1  # number of channels (wavelengths)

    # Setup extinction ratio
    inverse_ext_ratio_lin = math.pow(10, -ext_ratio / 10)
    gamma = (1 - math.sqrt(inverse_ext_ratio_lin)) / (1 + math.sqrt(inverse_ext_ratio_lin))  # [1]

    # Signal must be real
    sig = np.real(sig)

    if pp == 1:  # push-push
        phi_upper = math.pi * (sig + bias_upper * voltage_pi) / voltage_pi
        phi_lower = math.pi * (sig + bias_lower * voltage_pi) / voltage_pi
    else:  # push-pull
        phi_upper = math.pi / 2 * (sig + bias_upper * voltage_pi) / voltage_pi
        phi_lower = -math.pi / 2 * (sig + bias_lower * voltage_pi) / voltage_pi

    if np.shape(lightwave.field)[1] == 2 * np.size(lightwave.lam):  # dual polarization
        num_pol = 2
    else:
        num_pol = 1

    # Now set polarizations in alternate way (if they exist)
    num_cols = np.arange(num_pol * (num_chl - 1), num_pol * num_chl)
    # E.g. num_pol = 2, num_chl = 3 -> num_cols = [5 6]
    # E.g. num_pol = 1, num_chl = 3 -> num_cols = 3

    # Now modulation only on the existing polarizations
    for m in range(num_pol):
        pol = num_cols[m]
        if np.any(lightwave.field[:, pol]):
            temp = np.ravel((mathutils.fast_exp(phi_upper) + gamma * mathutils.fast_exp(phi_lower)) / (1 + gamma))
            temp = np.real(temp)
            lightwave.field[:, pol] = norm_factor * lightwave.field[:, pol] * temp
            # with default values -> E.field = E.field .* sin(sig)

    return lightwave
