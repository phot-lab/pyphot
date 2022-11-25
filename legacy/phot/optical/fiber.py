import numpy as np
import math

from ... import phot
from deprecated.phot.values import constants, globals
from deprecated.phot.utils import logger, mathutils
from .lightwave import Lightwave
from typing import Tuple


class Linear:
    def __init__(self):
        self.matin = None
        self.db0 = None
        self.is_unique = None
        self.is_scalar = None


class Fiber:
    _DEF_PLATES = 100  # number of waveplates when birefringence is on
    _DEF_BEAT_LENGTH = 20  # Default beat length [m]
    _DEF_STEP_UPDATE = 'cle'  # Default step-updating rule
    _DEF_IS_SYMMETRIC = False  # Default step computation
    _DEF_PHIFWM_CLE = 20  # Default X.dphimax [rad] for CLE stepupd rule
    _DEF_PHIFWM_NLP = 4  # Default X.dphimax [rad] for NLP & X.dphi1fwm
    _DEF_PHI_NLP = 0.003  # Default X.dphimax [rad] for NLP & ~X.dphi1fwm

    def __init__(self,
                 length: int = 10000,
                 lam: int = 1550,
                 alpha_b: float = 1,
                 dispersion: float = 17,
                 slope: float = 0,
                 n2: float = 0,
                 eff_area: float = 80,
                 coupling: str = None
                 ) -> None:
        """

        :param length: Fiber length [m]
        :param lam: Wavelength [nm] of fiber parameters
        :param alpha_b: Attenuation [dB/km]
        :param dispersion: Dispersion [ps/nm/km].
        :param slope: Slope [ps/nm^2/km]
        :param n2: Nonlinear index n2 [m^2/W]
        :param eff_area: Effective area [um^2]
        """
        self.length = length
        self.lam = lam
        self.alpha_b = alpha_b
        self.dispersion = dispersion
        self.slope = slope
        self.n2 = n2
        self.eff_area = eff_area
        self.coupling = coupling

        self.is_manakov = None
        self.pmd_param = None
        self.beat_length = None
        self.num_plates = None
        self.step_type = None
        self.linear = None
        self.dz_max = None  # maximum step
        self.dphi1_fwm = True  # FWM criterion for the first step setup.
        self.step_update = None

        self._is_unique = None
        self._dphi_max = None
        self._bandwidth = None
        self._alpha_linear = None
        self._gpu = False

    def transmit(self, lightwave: Lightwave, gpu: bool = False):
        self._gpu = gpu
        if np.shape(lightwave.field)[1] == 2 * np.size(lightwave.lam):  # dual polarization
            self._isy = True
        else:
            self._isy = False

        if np.size(lightwave.lam) > 1:
            self._is_unique = False  # separate-field propagation
        else:
            self._is_unique = True  # unique-field propagation

        # Check if Kerr effect is active
        if math.isinf(self.eff_area) or self.n2 == 0:  # no Kerr
            self._is_kerr = False
        else:
            self._is_kerr = True

        # Setup coupling
        self._setup_coupling()

        # SSFM check
        self._ssfm_check()

        # Setup parameters
        self._setup_params(lightwave)

        # SSFM Propagation
        first_dz, num_steps, lightwave = self._ssfm(lightwave, self.linear, self._betat)

        return lightwave, num_steps, first_dz

    def _setup_coupling(self):
        if self.coupling is not None:
            if self.coupling != "none" and self.coupling != "pol":
                raise RuntimeError("Coupling must be ''none'' or ''pol''.")

        if not self._isy:  # scalar propagation
            if self.is_manakov is not None and self.is_manakov:
                logger.warning("Cannot use Manakov equation in scalar propagation: forced to NLSE.")
            self.is_manakov = False  # because of no birefringence in scalar case
            if self.coupling is not None and self.coupling != "none":
                logger.warning("coupling not possible in scalar propagation")
            if self.pmd_param is not None and self.pmd_param != 0:
                logger.warning("PMD does not exist in scalar propagation: set to 0.")
            self.pmd_param = 0
            self.beat_length = 0
            self.coupling = "none"  # coupling impossible with one pol
        else:  # dual polarization
            if self.pmd_param is None:
                raise RuntimeError("Missing fiber PMD parameter [ps/sqrt(km)].")
            if self.is_manakov is not None and self.is_manakov:
                if self.pmd_param == 0:
                    # Note: coupling indeed makes a difference for the Kerr effect of CNLSE
                    self.coupling = "none"
                else:
                    if self.coupling is not None and self.coupling == "none":
                        logger.warning("No coupling but PMD ~= 0.")
                    else:
                        self.coupling = "pol"
            else:  # CNLSE
                self.is_manakov = False
                if self.coupling is None:
                    if not self._is_kerr and self.pmd_param == 0:
                        self.coupling = "none"
                    else:
                        self.coupling = "pol"
        if self.coupling == "pol":  # X-Y coupling
            if self.beat_length is None:
                self.beat_length = Fiber._DEF_BEAT_LENGTH  # default value
            if self.num_plates is None:
                self.num_plates = Fiber._DEF_PLATES  # default value
            len_corr = self.length / self.num_plates  # waveplate length [m] (aka correlation length)

            diff_group_delay = self.pmd_param / math.sqrt(len_corr) * math.sqrt(3 * math.pi / 8) / math.sqrt(
                1000) * 1e-3  # Differential
            # group delay (DGD) per unit length [ns/m] @ x.lambda within a
            # waveplate. To get [Df00] remember that within a waveplate the delay is dgd1*lcorr.
            if self.linear is None:
                linear = Linear()
                linear.matin = np.zeros((2, 2, self.num_plates))
                linear.db0 = np.zeros((self.num_plates, 2))
                for i in range(self.num_plates):  # SVD, hence different with the old FIBER version
                    linear.matin[:, :, i], linear.db0[i, :] = self._eigen_dec()
                # linear.db0 extended later
                linear.is_scalar = False
                self.linear = linear
        else:
            diff_group_delay = 0  # turn off all polarization and birefringence effects
            self.num_plates = 1
            self.coupling = "none"
            linear = Linear()
            linear.matin = 1  # scalar linear effects
            linear.db0 = 0
            linear.is_scalar = True
            self.linear = linear
        self.linear.is_unique = self._is_unique
        self._diff_group_delay = diff_group_delay

    def _ssfm_check(self):
        if self.dz_max is None or self.dz_max > self.length:
            self.dz_max = self.length  # maximum step

        # step update
        if self._is_kerr:
            if self.step_update is not None:
                if self.step_update != "cle" and self.step_update != "nlp":
                    raise RuntimeError("Unknown step update rule")
            else:
                self.step_update = Fiber._DEF_STEP_UPDATE  # default

            if self.step_update == "cle":
                if not self.dphi1_fwm:
                    raise RuntimeError("the combination X.dphi1fwm=false and X.stepupd=''cle'' is not possible")
                self._is_cle = True
            else:
                self._is_cle = False

            # first step parameter value
            if self._dphi_max is None:
                if self.dphi1_fwm:
                    if self._is_cle:
                        self._dphi_max = Fiber._DEF_PHIFWM_CLE  # CLE
                    else:
                        self._dphi_max = Fiber._DEF_PHIFWM_NLP  # NLP + PhiFWM in 1st step
                    if self._bandwidth is None:
                        self._dphi_max = self._dphi_max * 2.25  # because in [Mus18] they used the signal bandwidth,
                        # x1.5 smaller than the simulation bandwidth.
                else:
                    self._dphi_max = Fiber._DEF_PHI_NLP  # NLP

        # step type
        if self.step_type is not None:
            if self.step_type == "asymm":
                self._is_symmetric = False
            elif self.step_type == "symm":
                self._is_symmetric = True
            else:
                raise RuntimeError("Wrong step type")
        else:
            self._is_symmetric = Fiber._DEF_IS_SYMMETRIC

    def _setup_params(self, lightwave: Lightwave):
        self._alpha_linear = (math.log(10) * 1e-4) * self.alpha_b  # [m^-1]

        # Linear Parameters
        omega = 2 * math.pi * np.reshape(globals.FN, (-1, 1))  # angular frequency [rad/ns]
        b10 = 0  # retarded time frame
        b20 = -(np.square(
            self.lam)) / 2 / math.pi / constants.LIGHT_SPEED * self.dispersion * 1e-6  # beta2 [ns^2/m] @ x.lambda
        b30 = np.square((self.lam / 2 / math.pi / constants.LIGHT_SPEED)) * (
                2 * self.lam * self.dispersion + np.square(self.lam) * self.slope) * 1e-6
        betat = np.zeros(np.shape(lightwave.field))

        d_omega_i0 = 2 * math.pi * constants.LIGHT_SPEED * (1 / lightwave.lam - 1 / self.lam)  # [1/ns]
        ipm = 0
        if self._is_unique:  # unique field
            beta1 = b10 + self._diff_group_delay / 2  # [ns/m] @ E.lambda
            beta2 = b20 + b30 * d_omega_i0  # beta2 [ns^2/m] @ E.lambda
            beta3 = b30  # [ns^3/m] @ E.lambda

            betat[:, ipm] = np.ravel(omega * (beta1 + omega * (beta2 / 2 + omega * beta3 / 6)), order='F')
            # betat: deterministic beta coefficient [1/m]
            if self._isy:  # Add DGD on polarizations
                column = np.reshape(betat[:, ipm], (-1, 1)) - self._diff_group_delay / 2 * omega
                betat = np.append(betat, column, axis=1)
        else:  # separate-field
            freq = constants.LIGHT_SPEED / lightwave.lam  # carrier frequencies [GHz]
            max_freq = np.max(freq)  # [GHz]
            min_freq = np.min(freq)  # [GHz]
            central_freq = (max_freq + min_freq) / 2  # central frequency [GHz] (corresponding to the zero
            # frequency of the lowpass equivalent signal by convention)

            central_lam = constants.LIGHT_SPEED / central_freq  # central wavelength [nm]
            d_omega_ic = 2 * math.pi * constants.LIGHT_SPEED * (
                    1 / lightwave.lam - 1 / central_lam)  # Domega [1/ns] channel i vs central lambda
            d_omega_c0 = 2 * math.pi * constants.LIGHT_SPEED * (
                    1 / central_lam - 1 / self.lam)  # Domega [1/ns] central lambda vs lambda fiber parameters
            b1 = b10 + b20 * d_omega_ic + 0.5 * b30 * (
                    np.square(d_omega_i0) - np.square(d_omega_c0))  # ch's beta1 [ns/m]
            beta1 = b1 + self._diff_group_delay / 2  # [ns/m] @ GSTATE.LAMBDA
            beta2 = b20 + b30 * d_omega_i0  # beta2 [ns^2/m]@ E.lambda

            for i in range(np.size(lightwave.lam)):
                betat[:, i] = np.dot(omega, beta1.flat[i]) + 0.5 * np.square(
                    omega) * beta2.flat[i] + np.power(omega, 3).dot(b30) / 6  # beta coefficient [1/m]
                # Unfinished yet, Optilux code line: 446

        # GPU codes unfinished
        # if isa(E.field,'gpuArray')
        #     betat = gpuArray(betat); % put in GPU even betat to speed up
        # end

        # Nonlinear Parameters
        if not self._is_kerr:
            gam = 0
        else:
            gam = 2 * math.pi * self.n2 / (lightwave.lam * self.eff_area) * 1e18  # nonlinear coeff [1/mW/m]
            if math.isinf(gam):
                raise RuntimeError("Cannot continue: not finite nonlinear Kerr coefficient.")
        # x.aeff(1) is correct, because gam is normalized to fundamental mode [Ant16,Mum13].

        if self.is_manakov:  # Manakov
            if self._isy:
                nl_mat_coeff = 8 / 9
            else:
                nl_mat_coeff = 1  # Not a Manakov actually, but same model
        else:  # CNLSE
            nl_mat_coeff = 1
        self._gam = nl_mat_coeff * gam  # [1/mW/m], including Manakov correction, if active

        if (self.dispersion == 0 and self.slope == 0) or np.any(self._gam == 0):  # only GVD or only Kerr
            self._dphi_max = math.inf
            self.dz_max = self.length

        self._betat = betat

    def _ssfm(self, lightwave: Lightwave, linear: Linear, betat) -> Tuple[float, int, Lightwave]:
        # GPU codes unfinished
        # if isa(E.field,'gpuArray')
        #     E.field = gpuArray(E.field);
        # end

        num_steps = 1  # number of steps
        len_corr = self.length / self.num_plates  # waveplate length [m]

        dz, self._dphi_max = self._first_step(lightwave.field)
        half_alpha = 0.5 * self._alpha_linear  # [1/m]
        first_dz = dz
        z_prop = dz  # running distance [m]
        if self._is_symmetric:  # first half LIN step outside the cycle
            dzb, n_index = self._check_step(dz / 2, dz / 2, len_corr)
            lightwave.field = self._linear_step(linear, betat, dzb, n_index, lightwave.field)  # Linear step
        dzs = 0  # symmetric step contribution.
        while z_prop < self.length:  # all steps except the last
            # Nonlinear step
            lightwave.field = self._nonlinear_step(lightwave.field, dz)

            # Linear step 1/2: attenuation (scalar)
            lightwave.field = lightwave.field * np.exp(-half_alpha * dz)

            if self._is_symmetric:
                dzs = self._next_step(lightwave.field, dz)
                if (z_prop + dzs) > self.length:  # needed in case of last step
                    dzs = self.length - z_prop
                half_linear = (dz + dzs) / 2
            else:
                half_linear = dz

            # Linear step 2/2: GVD + birefringence
            dzb, n_index = self._check_step(z_prop + dzs / 2, half_linear, len_corr)  # zprop+dzs/2: end of step

            lightwave.field = self._linear_step(linear, betat, dzb, n_index, lightwave.field)  # Linear step

            if self._is_symmetric:
                dz, dzs = dzs, dz  # exchange dz and dzs
            else:
                dz = self._next_step(lightwave.field, dz)
            z_prop = z_prop + dz
            num_steps += 1
        last_step = self.length - z_prop + dz  # last step
        z_prop = z_prop - dz + last_step

        if self._is_symmetric:
            half_linear = last_step / 2
        else:
            half_linear = last_step

        # Last Nonlinear step
        if self._gam != 0:
            lightwave.field = self._nonlinear_step(lightwave.field, last_step)

        lightwave.field = lightwave.field * np.exp(-half_alpha * last_step)

        # Last Linear step: GVD + birefringence
        dzb, n_index = self._check_step(self.length, half_linear, len_corr)

        lightwave.field = self._linear_step(linear, betat, dzb, n_index, lightwave.field)  # last LIN

        return first_dz, num_steps, lightwave

    def _first_step(self, field: np.ndarray):
        if self.length == self.dz_max:
            step = self.dz_max
            phi_max = math.inf
        else:
            if self.dphi1_fwm:
                if self._bandwidth is None:
                    self._bandwidth = globals.SAMP_FREQ
                spac = self._bandwidth * np.square(self.lam) / constants.LIGHT_SPEED  # bandwidth in [nm]
                if self._is_unique:  # min: worst case among spatial modes
                    step = np.min(self._dphi_max / np.abs(self.dispersion) / (
                            2 * math.pi * spac * self._bandwidth * 1e-3) * 1e3)  # [m]
                else:  # separate fields: FWM bandwidth is substituted by walk-off bandwidth
                    step = np.min(self._dphi_max / np.abs(self.dispersion) / (
                            2 * math.pi * spac @ self._bandwidth * 1e-3) * 1e3)  # [m]
                if step > self.dz_max:
                    step = self.dz_max
                if self.step_update == "nlp":  # nonlinear phase criterion
                    max_gam = np.max(self._gam)  # CNLSE
                    inv_lnl = np.max(np.square(np.abs(field))) * max_gam  # max of 1/Lnl [1/m]
                    if self._alpha_linear == 0:
                        leff = step
                    else:
                        leff = (1 - np.exp(-self._alpha_linear * step)) / self._alpha_linear
                    phi_max = inv_lnl * leff  # recalculate max nonlinear phase [rad] per step
                else:
                    phi_max = self._dphi_max
            else:  # nonlinear phase criterion
                step = self._next_step(field, np.nan)
                phi_max = self._dphi_max

        return step, phi_max

    def _next_step(self, field: np.ndarray, dz_old):
        if self._is_cle:  # constant local error (CLE)
            if self._is_symmetric:  # See [Zha05,Zha08]
                q = 3
            else:
                q = 2
            step = dz_old * np.exp(self._alpha_linear / q * dz_old)  # [m]
        else:  # nonlinear phase criterion
            if self._isy:  # max over time
                phase_max = np.max(np.square(np.abs(field[:, 0::2])) + np.square(np.abs(field[:, 1::2])))
            else:
                phase_max = np.max(np.square(np.abs(field)))
            inv_lnl = np.max(phase_max * self._gam)  # max over channels
            len_eff = self._dphi_max / np.max(inv_lnl)  # effective length [m] of the step
            dl = self._alpha_linear * len_eff  # ratio effective length/attenuation length
            if dl >= 1:
                step = self.dz_max
            else:
                if self._alpha_linear == 0:
                    step = len_eff
                else:
                    step = -1 / self._alpha_linear * np.log(1 - dl)
        if step > self.dz_max:
            dz = self.dz_max
        else:
            dz = step
        return dz

    def _check_step(self, z_prop, dz, len_corr: float):
        z_ini = z_prop - dz  # starting coordinate of the step [m]
        z_end = z_prop  # ending coordinate of the step [m]
        # First waveplate index is 1
        n_ini = np.floor(z_ini / len_corr)  # waveplate of starting coordinate
        n_end = np.ceil(z_end / len_corr)  # waveplate of ending coordinate

        n_mid = np.arange(n_ini, n_end)  # waveplate indexes
        if n_ini == n_end - 1:  # start/end of the step within a waveplate
            dz_split = dz
            num_trunk = np.array([0, 0])
            #     if isa(nini,'gpuArray')
            #         ntrunk = gpuArray(ntrunk);
            #     end
        else:  # multi-waveplate step
            z_mid = len_corr * n_mid[0:-1]  # waveplate mid-coordinates
            dz_split = np.concatenate(([z_mid[0] - z_ini], np.diff(z_mid), [z_end - z_mid[-1]]), axis=1)
            num_trunk = np.repeat(n_mid, 2).reshape(-1, 1)
        if (n_ini - 1) * len_corr != z_ini:
            num_trunk.flat[0] = 0  # starting coord. over a trunk
        if (n_end - 1) * len_corr != z_end:
            num_trunk.flat[-1] = 0  # ending coord. over a trunk

        if np.isscalar(dz_split):
            if dz_split != 0:
                dzb = dz_split
                n_index = n_mid[0]
            else:
                dzb = np.array([])
                n_index = np.array([])
        else:
            dzb = dz_split[dz_split != 0]  # remove zero-length steps
            n_index = n_mid[dz_split != 0]  # remove zero-length steps
        return dzb, n_index

    def _linear_step(self, linear: Linear, betat, dzb, n_index, field: np.ndarray) -> np.ndarray:
        if self._gpu:
            field = phot.to_gpu(field)
        field = np.fft.fft(field, axis=0)
        for nt in range(np.size(dzb)):  # the step is made of multi-waveplates
            if np.isscalar(n_index):
                index = int(n_index)
            else:
                index = int(n_index[nt])

            if linear.is_unique:
                if linear.is_scalar:
                    field = field * np.conj(linear.matin)
                    temp = mathutils.fast_exp(-(betat + linear.db0) * dzb)
                    if self._gpu:
                        temp = phot.to_gpu(temp)
                    field = field * temp
                    field = field * linear.matin
                else:
                    field = field @ np.conj(linear.matin[:, :, index])  # apply unitary matrix
                    field = field * mathutils.fast_exp(-(betat + linear.db0[index, :]) * dzb[nt])
                    field = field @ linear.matin[:, :, index].T  # return in the reference system
            else:
                field = _fast_matmul(field, np.conj(linear.matin[:, :, index]))  # apply unitary matrix
                field = field * mathutils.fast_exp(-(betat + linear.db0[index, :]) * dzb[nt])
                field = _fast_matmul(field, linear.matin[:, :, index].T)  # return in the reference system
        field = np.fft.ifft(field, axis=0)
        if self._gpu:
            field = phot.to_cpu(field)
        return field

    def _nonlinear_step(self, field: np.ndarray, dz: float) -> np.ndarray:
        if self._alpha_linear == 0:
            len_eff = dz
        else:
            len_eff = (1 - np.exp(-self._alpha_linear * dz)) / self._alpha_linear  # effective length [m] of dz
        gam_len_eff = self._gam * len_eff  # [1/mW]
        if self._is_unique:  # Unique Field
            phi = np.sum(np.square(np.abs(field)), axis=1) * gam_len_eff  # nl phase [rad].
            exp_phi = mathutils.fast_exp(-phi)
            field = exp_phi * field
            if not self.is_manakov and np.shape(field)[1] > 1:  # CNLSE, only in dual-polarization
                stroke = 2 * (np.real(field[:, 0]) * np.imag(field[:, 1]) - np.imag(field[:, 0]) * np.real(field[:, 1])
                              ) * gam_len_eff[0]  # stokes comp. #3
                exp_gam_len_eff = mathutils.fast_exp(stroke / 3)
                cos_phi = np.real(exp_gam_len_eff)  # (fast) cos(gamleff*s3/3)
                sin_phi = np.imag(exp_gam_len_eff)  # (fast) sin(gamleff*s3/3)
                uxx = cos_phi * field[:, 0] + sin_phi * field[:, 1]
                uyy = -sin_phi * field[:, 0] + cos_phi * field[:, 1]
                field[:, 0] = uxx
                field[:, 1] = uyy
        else:  # separate-field
            # unfinished yet, optilux code line: 737
            pass
        return field

    def _eigen_dec(self):
        if self.coupling == "none":
            u = np.eye(2)
            s = np.zeros((1, 2))  # eigenvalues
        elif self.coupling == "pol":  # couple only polarizations
            u = mathutils.rand_unitary(3)  # random unitary matrix (Haar matrix)
            q = math.sqrt(math.pow(math.pi, 3) / 2) / self.beat_length * np.random.randn(1, 3)  # [1/m]
            w = np.sqrt(np.sum(np.square(q), axis=1))  # Maxwellian distribution. deltabeta0 [1/m]
            s = np.concatenate((w, -w), axis=1) / 2
        else:
            raise RuntimeError("Unknown coupling method.")
        return u, s


def _fast_matmul(array: np.ndarray, other) -> np.ndarray:
    """
    https://stackoverflow.com/questions/40508500/how-to-dynamically-reshape-matrix-block-wise
    :param array: first matrix
    :param other: the other one
    :return: matrix after matrix multiplication
    """
    if not np.isscalar(other):
        rows, cols = np.shape(array)
        array = np.reshape(np.transpose(np.reshape(array, (rows, 2, -1), order='F'), (0, 2, 1)),
                           (-1, 2), order='F')  # 2: polarizations
        array = array @ other
        array = np.reshape(np.transpose(np.reshape(array, (rows, int(cols / 2), -1), order='F'), (0, 2, 1)),
                           (rows, -1), order='F')
    return array
