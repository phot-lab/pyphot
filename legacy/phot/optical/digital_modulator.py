from deprecated.phot.values import globals
from deprecated.phot.utils import mathutils, logger
from deprecated.phot.optical import signal
from . import format
from .sample import seq2samp
import numpy as np
import math


class DigitalModulator:
    def __init__(self, mod_format: str = None,
                 pulse_type: str = None,
                 rolloff: float = 0,
                 pre_emphasis: str = "asin",
                 norm: str = "iid",
                 bandwidth=None,
                 duty=1
                 ):
        self.pre_emphasis = pre_emphasis  # digital pre-emphasis type
        self.bandwidth = bandwidth
        self.rolloff = rolloff  # pulse roll-off
        self.mod_format = mod_format  # modulation format
        self.duty = duty
        self.param = 0
        self.norm = norm
        self.num_samp = None
        self.fir_taps = None
        self.num_fft = globals.NUM_SAMP
        if pulse_type != "rc" and pulse_type != "rootrc" and pulse_type != "userfir" and pulse_type != "dirac" and pulse_type != "costails" and pulse_type != "rect":
            if self.bandwidth is None:
                raise RuntimeError("Missing filter bandwidth for pulse shaping")
            signal.my_filter(pulse_type, 1, self.bandwidth, self.param)

        num_tini = globals.SAMP_FREQ / globals._sym_rate  # Wished samples per symbol
        nt, nd = mathutils.rat(num_tini)  # oversample, then down sample at the end

        if self.num_samp is not None:
            if math.floor(self.num_samp) != self.num_samp:
                raise RuntimeError("The number of samples per symbol must be an integer.")
            num_samp = self.num_samp  # the function works with Nsps samples per symbol till
            # the final resampling emulating the DAC
        else:
            num_samp = nt

        if nt / num_tini > 10:
            logger.warning(
                'resampling may take big times/memory. Consider using sampling '
                'or symbol rates such that \n[N,D]=rat(GSTATE.FSAMPLING/symbrate) '
                'contains a small integer N (now N={}, D={}). See also inigstate.m.'.format(nt, nd))

        self.pulse_type = pulse_type
        self.num_tini = num_tini
        self.num_samp = num_samp
        self.nd = nd
        self.nt = nt

    def modulate(self, seq):
        """
        :param seq: Random number sequence
        :return: Electric source and normalization factor
        """

        num_sym = np.size(seq)
        num_sym_up_down = math.ceil((self.num_fft - 1) / self.num_tini)
        if num_sym < num_sym_up_down:
            raise RuntimeError('Too few symbols to fill the required number of samples.')
        elif num_sym > num_sym_up_down:
            num_sym = num_sym_up_down
            logger.warning('Too many symbols. Sequence truncated to {} symbols.'.format(num_sym))

        # 1. convert the pattern into stars of the constellations
        level = seq2samp(seq, self.mod_format)

        # 2. create a linearly modulated digital signal
        elec = self._elec_source(level, self.pulse_type, num_sym, self.num_samp, self.nd, self.num_fft)

        # 3. resample if necessary
        if self.nt != self.num_samp:
            # Apply DAC (Note: resample.m generates border effects. interp is better)
            if np.mod(self.nt, self.num_samp) == 0:
                # The result of this function is a little different with MATLAB's interp
                elec = signal.interp(elec, self.nt / self.num_samp)
            else:
                # Unfinished here, Optilux calls circresample
                # elec = circresample(elec,Nt,Nsps); % (circular) resample
                pass

        # 4. Perform pre-emphasis
        if self.pre_emphasis is not None:
            norm_factor = max(np.max(np.abs(np.real(elec))), np.max(np.abs(np.imag(elec))))
            if self.pre_emphasis == "asin":
                # Thanks to normf, the result of each asin is for sure real, as required by a M.ach-Zehnder modulator.
                # However, to preserve the energy, the Mach-Zehnder must know such a normalization factor.
                elec = np.arcsin(np.real(elec) / norm_factor) + 1j * np.arcsin(np.imag(elec) / norm_factor)
        else:
            norm_factor = 1

        return elec, norm_factor

    def _elec_source(self, ak, pulse_type, num_sym, num_samp, nd, num_fft):
        # The idea is the following: the sequence is first up-sampled to tx_param.num_samp
        # samples per symbol, and then filtered to create the PAM signal.

        if pulse_type == "rc" or pulse_type == "rootrc" or pulse_type == "costails":
            if self.rolloff is None:
                raise RuntimeError("Undefined rolloff")
            if self.rolloff > 1 or self.rolloff < 0:
                raise RuntimeError("The roll-off must be 0<= roll-off <= 1")

        if self.duty is None:
            self.duty = 1
        elif self.duty <= 0 or self.duty > 1:
            raise RuntimeError("must be 0 < duty <= 1 ")
        if pulse_type != "rc" and pulse_type != "rootrc" and pulse_type != "userfir" and pulse_type != "dirac" and pulse_type != "rect" and pulse_type != "costails":
            flag = True  # the pulse is the impulse response of my filter
            if self.param is None:
                self.param = 0
            if self.bandwidth is None:
                raise RuntimeError("Missing bandwidth for my filter")
        else:
            flag = False

        # Modulate
        if pulse_type == "userfir":
            if self.fir_taps is None:
                raise RuntimeError("Missing FIR filter taps")
            else:
                if np.size(self.fir_taps) > (num_sym * num_samp):
                    raise RuntimeError("Too many taps for FIR")
                if not np.isrealobj(self.fir_taps):
                    raise RuntimeError("FIR taps must be real.")
                self.fir_taps = np.reshape(self.fir_taps, (-1, 1), order='F')

        aku = signal.up_sample(ak, num_samp)
        aku = aku.ravel(order='F')
        aku = np.fft.fft(aku[0:num_samp * num_sym])  # truncate if necessary
        if flag:  # filter the signal
            ffir = self.duty * np.fft.fftshift(np.arange(-num_samp / 2, num_sym / 2, 1 / num_sym))
            hfir = signal.my_filter(pulse_type, ffir.reshape(-1, 1), self.bandwidth, self.param)
        else:
            el_pulse = self._pulse_design(pulse_type, num_samp, num_sym)  # single pulse
            hfir = np.fft.fft(np.fft.fftshift(el_pulse), axis=0)
            if pulse_type == "rootrc":  # square-root raised cosine
                hfir = np.sqrt(hfir * num_samp)
                # *Nt: because I'm using filters normalized in peak spectrum (as if symbol time was 1)

        aku = np.reshape(aku, (-1, 1))
        elec = np.fft.ifft(aku * hfir, axis=0)  # create PAM signal

        if np.size(elec) < num_fft:
            raise RuntimeError(
                "It is impossible to get the desired number of samples with the given pattern and sampling rate")
        elif np.size(elec) > num_fft:
            elec = elec[np.ceil(np.arange(0, num_fft * nd, nd))]
        # normalize to unit power
        if self.norm == "iid":  # i.i.d. symbols
            # See [3], power spectra of linearly modulated signals
            format_info = format.get_format_info(self.mod_format)
            var_ak = format_info.sym_var  # expected variance
            mean_ak = format_info.sym_mean  # expected value or mean
            avg = (var_ak * np.sum(np.square(np.abs(hfir)))) / num_sym + np.square(np.abs(mean_ak)) * np.sum(
                np.square(np.abs(hfir[0::num_sym]))) / np.square(num_samp)
        elif self.norm == "mean":
            avg = np.mean(np.square(np.abs(elec)))
        elif self.norm == "no":
            avg = 1
        else:
            raise RuntimeError("Unknwon normalization method")
        return elec / math.sqrt(avg)

    def _pulse_design(self, pulse_type, num_samp, num_sym):
        """
        PULSEDESIGN Creates the fundamental pulse
        Y=PULSEDESIGN(PTYPE,NSPS,NSYMB,PAR) returns in the vector [NSYMB*NSPS,1]
        the fundamental pulse whose type is defined in PTYPE.
        :param pulse_type:
        :param num_samp:
        :param num_sym:
        :return:
        """

        el_pulse = np.zeros((num_samp * num_sym, 1))  # Electric pulse
        if pulse_type == "rc" or pulse_type == "rootrc":
            tfir = (1 / self.duty) * np.arange(-num_sym / 2, num_sym / 2, 1 / num_samp)
            tfir = np.reshape(tfir, (-1, 1))
            el_pulse = np.sinc(tfir) * np.cos((math.pi * tfir * self.rolloff)) / (
                    1 - np.square(2 * self.rolloff * tfir))
            el_pulse[np.logical_not(np.isfinite(el_pulse))] = self.rolloff / 2 * np.sin(
                math.pi / 2 / self.rolloff)  # Avoid NaN
        elif pulse_type == "costails":
            nl = round(0.5 * (1 - self.rolloff) * self.duty * num_samp)  # start index of cos roll-off
            nr = self.duty * num_samp - nl  # end index of cos roll-off

            num_mark = np.arange(0, nl)  # indices where the pulse is 1
            num_cos = np.arange(nl, nr)  # transition region of cos roll-off

            el_pulse[num_samp * num_sym / 2 + num_mark] = 1
            h_period = self.duty * num_samp - 2 * nl
            if h_period != 0:
                el_pulse[num_cos + num_samp * num_sym / 2] = 0.5 * (
                        1 + np.cos(math.pi / h_period * (num_cos - nl + 0.5)))
            el_pulse[0: num_samp * num_sym / 2] = np.flipud(
                el_pulse[num_samp * num_sym / 2: num_samp * num_sym])  # first half of the pulse
        else:
            raise RuntimeError("The pulse ptype does not exist")
        return el_pulse
