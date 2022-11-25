import numpy as np
from deprecated.phot.values import globals
from deprecated.phot.utils import mathutils
from .lightwave import Lightwave
import math


class LaserSource:
    """
    Multi-channel laser source
    E = LASERSOURCE(PTX,LAM) creates a comb of constant waves of powers
    PTX [mW] at wavelengths LAM [nm]. Such waves are stored in the struct E of fields:
    """

    def __init__(self,
                 power,  # power in linear scale [mW]
                 lam: int = 1550,
                 num_pol: int = 1,
                 line_width=None,
                 n0: float = None
                 ) -> None:
        """
        Initialize parameters of laser source
        :param num_pol: polarization schema. 'single' only the x-polarization is created,
        otherwise even the y-polarization is created and set equal to zero in absence of noise.
        :param n0: the one-sided spectral density [dB/GHz] of a Gaussian complex noise added to the laser field.
        line_width: It can be a scalar or a vector of the same length of the wavelengths.

        A Wiener phase noise with such a line_width is added to the phase of Lightwave.
        """

        self.num_power = np.size(power)  # Number of the power of transmit channel
        self.num_carr = np.size(lam)  # Number of carriers
        self.num_pol = num_pol
        self.lam = lam
        self.n0 = n0
        if self.num_power > 1 and self.num_power != self.num_carr:
            raise RuntimeError('Powers and wavelengths must have the same length')

        if line_width is not None:
            if np.isscalar(line_width):
                self.line_width = line_width * np.ones((1, self.num_carr))
            elif line_width.size != self.num_carr:
                raise RuntimeError('The line_width must have the same length of the number of wavelengths.')
        else:
            self.line_width = 0

        if self.num_power == 1:
            if np.isscalar(power):
                self.power = np.full(self.num_carr, power)
            else:
                self.power = np.full(self.num_carr, power[0])
        else:
            self.power = power

    def gen_light(self) -> Lightwave:
        num_samp = globals.NUM_SAMP

        # uniformly spaced carriers
        field = np.zeros((num_samp, self.num_carr * self.num_pol))
        lightwave = Lightwave(self.lam, field)

        # by default, fully polarized on x (odd columns)
        for i in range(self.num_carr):
            lightwave.field[:, i * self.num_pol] = np.full(num_samp, math.sqrt(self.power[i]))

        if np.any(self.line_width):
            # Add phase noise. This part referred to:
            # 'freq_noise  = (ones(Nsamp,1) * sqrt(2*pi*linewidth./GSTATE.FSAMPLING)) .* randn( Nsamp, Nch)' in Optilux.
            # New a vector named one(Nsamp,1) in equation above.
            ones_vec = np.ones((num_samp, 1))

            # New a matrix named randn( Nsamp, Nch), which follows standard normal distribution X(0,1).
            rand_mat = np.random.normal(size=(num_samp, self.num_carr))
            freq_noise = self.line_width * 2 * math.pi
            freq_noise /= globals.SAMP_FREQ
            freq_noise = np.sqrt(freq_noise)
            freq_noise = ones_vec @ freq_noise
            freq_noise *= rand_mat
            freq_noise[0][0] = 0
            phase_noise = freq_noise.cumsum(1)

            # Brownian bridge [Wiki]
            tim = np.arange(0, num_samp)
            tim = tim.reshape((-1, 1), order='F')
            phase_noise = phase_noise - (tim / (num_samp - 1)) @ (phase_noise[-1, :].reshape((1, -1), order='F'))
            phase_noise = mathutils.fast_exp(phase_noise)
            for i in range(self.num_carr):
                lightwave.field[:, i * self.num_pol] *= phase_noise[:, i * self.num_pol]

        # Add Gaussian complex white noise (ASE)
        if self.n0 is not None:
            n0_lin = math.pow(10, self.n0 / 10)
            sigma = math.sqrt(n0_lin / 2 * globals.SAMP_FREQ)  # noise std
            real_mat = np.random.normal(size=(num_samp, self.num_carr * self.num_pol))
            imag_mat = np.random.normal(size=(num_samp, self.num_carr * self.num_pol))
            lightwave.field = lightwave.field + sigma * (real_mat + imag_mat * 1j)

        return lightwave
