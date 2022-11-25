import numpy as np
from deprecated.phot.values import globals, constants
import math
from deprecated.phot.utils import mathutils
from .signal import my_filter
from .lightwave import Lightwave


class RxFrontend:
    def __init__(self, filter_type: str = None,
                 lam: int = 1550,
                 mod_format: str = None,
                 bandwidth=None,
                 mz_delay: int = 1,  # default interferometer delay: 1 symbol
                 optic_param=None  # # optical filter extra parameters
                 ):
        self.bandwidth = bandwidth  # OBPF bandwidth normalized to SYMBRATE
        self.filter_type = filter_type
        self.mod_format = mod_format
        self.optic_param = optic_param
        self.mz_delay = mz_delay  # interferometer delay
        self.disp_accum = None  # post compensating fiber accumulated dispersion [ps/nm].
        # The fiber is inserted before the optical filter.
        self.slope_accum = None  # post compensating fiber accumulated slope [ps/nm^2]

        if self.optic_param is None:
            self.optic_param = 0  # dummy value
        if self.mz_delay <= 0 or self.mz_delay > 1:
            raise RuntimeError("The delay of the interferometer must be  0 < mzdel <= 1")
        if self.mod_format is None:
            raise RuntimeError("Missing modulation format")

        self._lam = lam
        self._sym_rate = globals._sym_rate

    def receive(self, lightwave: Lightwave) -> np.ndarray:

        # Create linear optical filters: OBPF (+fiber)
        freq_norm = globals.FN / self._sym_rate
        if self.disp_accum is not None:
            hf = _post_fiber(self.disp_accum, self.slope_accum, self._lam, self._lam,
                             lightwave.lam)
            hf = hf * my_filter(self.filter_type, freq_norm, 0.5 * self.bandwidth,
                                self.optic_param)
        else:
            hf = my_filter(self.filter_type, freq_norm, 0.5 * self.bandwidth,
                           self.optic_param)

        # 1. apply optical filter
        lightwave = _filter_env(lightwave, self._lam, hf)

        # 2. optical to electrical conversion
        nt = globals.SAMP_FREQ / self._sym_rate  # number of points per symbol

        iric = self._optic2elec(lightwave, nt)

        return iric

    def _optic2elec(self, lightwave: Lightwave, num_pts) -> np.ndarray:
        num_fft = np.size(lightwave.field)
        if self.mod_format == "ook":
            iric = np.sum(np.square(np.abs(lightwave.field)), axis=1)  # PD. sum is over polarizations
        elif self.mod_format == "dpsk":
            delay = mathutils.n_mod(np.arange(num_fft) - round(self.mz_delay * num_pts),
                                    num_fft)  # interferometer delay

            iric = np.sum(np.real(lightwave.field * np.conj(lightwave.field[delay])), axis=1)  # MZI + PD
        elif self.mod_format == "dqpsk":
            delay = mathutils.n_mod(np.arange(num_fft) - round(self.mz_delay * num_pts),
                                    num_fft)  # interferometer delay
            iric_real = np.sum(
                mathutils.fast_exp(-math.pi / 4) * np.real(lightwave.field * np.conj(lightwave.field[delay])), axis=1)
            iric_imag = np.sum(
                mathutils.fast_exp(math.pi / 4) * np.real(lightwave.field * np.conj(lightwave.field[delay])), axis=1)
            iric = iric_real + iric_imag * 1j
        else:  # coherent detection
            iric = lightwave.field
        return iric


def _post_fiber(disp_acc, slope_acc, lam, lam_rx, lam_central) -> np.ndarray:
    LIGHT_SPEED = constants.LIGHT_SPEED  # [m/s]
    b20z = -lam ** 2 / 2 / math.pi / LIGHT_SPEED * disp_acc * 1e-3  # beta2*z [ns^2] @ lam0
    b30z = np.square((lam / 2 / math.pi / LIGHT_SPEED)) * (
            2 * lam * disp_acc + np.square(lam) * slope_acc) * 1e-3  # beta3*z [ns^3] @ lam0
    # lam:  wavelength of the channel's carrier under detection
    # lam_rx: wavelength @ rx parameters
    # lam_central: wavelength of central frequency of bandpass signal

    # d_omega_ik: [1/ns]. "i" -> at ch. i, "0" -> at lam0
    d_omega_i0 = 2 * math.pi * LIGHT_SPEED * (1 / lam_rx - 1 / lam)
    d_omega_ic = 2 * math.pi * LIGHT_SPEED * (1 / lam_rx - 1 / lam_central)
    d_omega_c0 = 2 * math.pi * LIGHT_SPEED * (1 / lam_central - 1 / lam)

    beta_1z = b20z * d_omega_ic + 0.5 * b30z * (np.square(d_omega_i0) - np.square(d_omega_c0))  # [ns]
    beta_2z = b20z + b30z * d_omega_i0  # beta2*z [ns^2]@ lam

    # dispersion of the channels
    omega = 2 * math.pi * np.conj(globals.FN.T)  # angular frequency [rad/ns]
    betat = omega * beta_1z + 0.5 * np.square(omega) * beta_2z + (omega ** 3) * b30z / 6

    hf = mathutils.fast_exp(-betat)
    return hf


def _filter_env(lightwave: Lightwave, lam, hf) -> Lightwave:
    freq_central = constants.LIGHT_SPEED / lightwave.lam  # central frequency [GHz] (
    # corresponding to the zero frequency of the lowpass equivalent signal by convention)
    freq = constants.LIGHT_SPEED / lam  # carrier frequency [GHz]
    delta_fn = freq_central - freq  # carrier frequency spacing [GHz]
    min_freq = globals.FN[1] - globals.FN[0]  # resolution [GHz]
    nd_fn = round(delta_fn / min_freq)  # spacing in points
    hf = np.reshape(hf, (-1, 1))

    lightwave.field = np.fft.fft(lightwave.field, axis=0)
    lightwave.field = np.roll(lightwave.field, nd_fn)  # undo what did in MULTIPLEXER
    lightwave.field = np.fft.ifft(hf * lightwave.field, axis=0)
    return lightwave
