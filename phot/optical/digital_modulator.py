from phot.values import globals
from phot.utils import mathutils, logger
from phot.optical import signal, format
import numpy as np
import math


class TxParam:
    """
    TxParam stands for transmit side parameter
    """

    def __init__(self, mod_format=None, pre_emphasis="asin", rolloff=0, norm="iid"):
        self.pre_emphasis = pre_emphasis  # digital pre-emphasis type
        self.norm = norm
        self.num_samp = None
        self.mod_format = mod_format  # modulation format
        self.param = 0
        self.bandwidth = None
        self.duty = 1
        self.rolloff = rolloff  # pulse roll-off
        self.fir_taps = None


class DigitalModulator:
    def __init__(self, mod_format, pulse_type, tx_param):
        self.num_fft = globals.NUM_SAMP
        if pulse_type != "rc" and pulse_type != "rootrc" and pulse_type != "userfir" and pulse_type != "dirac" and pulse_type != "costails" and pulse_type != "rect":
            if tx_param.bandwidth is None:
                raise RuntimeError("Missing filter bandwidth for pulse shaping")
            signal.my_filter(pulse_type, 1, tx_param.bandwidth, tx_param.param)
        tx_param.mod_format = mod_format

        num_tini = globals.SAMP_FREQ / globals._sym_rate  # Wished samples per symbol
        nt, nd = mathutils.rat(num_tini)  # oversample, then down sample at the end

        if tx_param.num_samp is not None:
            if math.floor(tx_param.num_samp) != tx_param.num_samp:
                raise RuntimeError("The number of samples per symbol must be an integer.")
            num_samp = tx_param.num_samp  # the function works with Nsps samples per symbol till
            # the final resampling emulating the DAC
        else:
            num_samp = nt

        if nt / num_tini > 10:
            logger.warning(
                'resampling may take big times/memory. Consider using sampling '
                'or symbol rates such that \n[N,D]=rat(GSTATE.FSAMPLING/symbrate) '
                'contains a small integer N (now N={}, D={}). See also inigstate.m.'.format(nt, nd))

        self.mod_format = mod_format
        self.pulse_type = pulse_type
        self.tx_param = tx_param
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
        elec = elec_source(level, self.pulse_type, self.tx_param, num_sym, self.num_samp, self.nd, self.num_fft)

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
        if self.tx_param.pre_emphasis is not None:
            norm_factor = max(np.max(np.abs(np.real(elec))), np.max(np.abs(np.imag(elec))))
            if self.tx_param.pre_emphasis == "asin":
                # Thanks to normf, the result of each asin is for sure real, as required by a M.ach-Zehnder modulator.
                # However, to preserve the energy, the Mach-Zehnder must know such a normalization factor.
                elec = np.arcsin(np.real(elec) / norm_factor) + 1j * np.arcsin(np.imag(elec) / norm_factor)
        else:
            norm_factor = 1

        return elec, norm_factor


def seq2samp(seq, mod_format):
    format_info = format.get_format_info(mod_format)
    num_bits = math.log2(format_info.digit)
    rows, cols = np.shape(seq)  # cols == 1: symbol. cols == 2: bits
    if mod_format == "randn":
        return seq
    seq = mathutils.dec2bin(seq, num_bits)

    # From now on, pat is a binary matrix
    m = math.pow(2, cols)  # constellation size

    if format_info.family == "ook":
        level = 2 * seq  # average energy: 1
    elif mod_format == "bpsk" or mod_format == "dpsk" or mod_format == "psbt" or (mod_format == "psk" and m == 2):
        level = 2 * seq - 1  # average energy: 1
    elif mod_format == "qpsk" or mod_format == "dqpsk":
        level = 2 * seq - 1  # drive iqmodulator with QPSK
        level = (level[:, 0] + level[:, 1] * 1j) / math.sqrt(2)  # average energy: 1
        level = np.reshape(level, (-1, 1))
    else:
        raise RuntimeError("Unknown modulation format")

    return level


def elec_source(ak, pulse_type, tx_param, num_sym, num_samp, nd, num_fft):
    # The idea is the following: the sequence is first up-sampled to tx_param.num_samp
    # samples per symbol, and then filtered to create the PAM signal.

    if pulse_type == "rc" or pulse_type == "rootrc" or pulse_type == "costails":
        if tx_param.rolloff is None:
            raise RuntimeError("Undefined rolloff")
        if tx_param.rolloff > 1 or tx_param.rolloff < 0:
            raise RuntimeError("The roll-off must be 0<= roll-off <= 1")

    if tx_param.duty is None:
        tx_param.duty = 1
    elif tx_param.duty <= 0 or tx_param.duty > 1:
        raise RuntimeError("must be 0 < duty <= 1 ")
    if pulse_type != "rc" and pulse_type != "rootrc" and pulse_type != "userfir" and pulse_type != "dirac" and pulse_type != "rect" and pulse_type != "costails":
        flag = True  # the pulse is the impulse response of my filter
        if tx_param.param is None:
            tx_param.param = 0
        if tx_param.bandwidth is None:
            raise RuntimeError("Missing bandwidth for my filter")
    else:
        flag = False

    # Modulate
    if pulse_type == "userfir":
        if tx_param.fir_taps is None:
            raise RuntimeError("Missing FIR filter taps")
        else:
            if np.size(tx_param.fir_taps) > (num_sym * num_samp):
                raise RuntimeError("Too many taps for FIR")
            if not np.isrealobj(tx_param.fir_taps):
                raise RuntimeError("FIR taps must be real.")
            tx_param.fir_taps = np.reshape(tx_param.fir_taps, (-1, 1), order='F')
    aku = signal.up_sample(ak, num_samp)
    aku = aku.ravel(order='F')
    aku = np.fft.fft(aku[0:num_samp * num_sym])  # truncate if necessary
    if flag:  # filter the signal
        ffir = tx_param.duty * np.fft.fftshift(np.arange(-num_samp / 2, num_sym / 2, 1 / num_sym))
        hfir = signal.my_filter(pulse_type, ffir.reshape(-1, 1), tx_param.bandwidth, tx_param.param)
    else:
        el_pulse = pulse_design(pulse_type, num_samp, num_sym, tx_param)  # single pulse
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
    if tx_param.norm == "iid":  # i.i.d. symbols
        # See [3], power spectra of linearly modulated signals
        format_info = format.get_format_info(tx_param.mod_format)
        var_ak = format_info.sym_var  # expected variance
        mean_ak = format_info.sym_mean  # expected value or mean
        avg = (var_ak * np.sum(np.square(np.abs(hfir)))) / num_sym + np.square(np.abs(mean_ak)) * np.sum(
            np.square(np.abs(hfir[0::num_sym]))) / np.square(num_samp)
    elif tx_param.norm == "mean":
        avg = np.mean(np.square(np.abs(elec)))
    elif tx_param.norm == "no":
        avg = 1
    else:
        raise RuntimeError("Unknwon normalization method")
    return elec / math.sqrt(avg)


def pulse_design(pulse_type, num_samp, num_sym, tx_param):
    """
    PULSEDESIGN Creates the fundamental pulse
    Y=PULSEDESIGN(PTYPE,NSPS,NSYMB,PAR) returns in the vector [NSYMB*NSPS,1]
    the fundamental pulse whose type is defined in PTYPE.
    :param pulse_type:
    :param num_samp:
    :param num_sym:
    :param tx_param: tx_param is a struct (see main help of DIGITALMOD).
    :return:
    """

    el_pulse = np.zeros((num_samp * num_sym, 1))  # Electric pulse
    if pulse_type == "rc" or pulse_type == "rootrc":
        tfir = (1 / tx_param.duty) * np.arange(-num_sym / 2, num_sym / 2, 1 / num_samp)
        tfir = np.reshape(tfir, (-1, 1))
        el_pulse = np.sinc(tfir) * np.cos((math.pi * tfir * tx_param.rolloff)) / (
                1 - np.square(2 * tx_param.rolloff * tfir))
        el_pulse[np.logical_not(np.isfinite(el_pulse))] = tx_param.rolloff / 2 * np.sin(
            math.pi / 2 / tx_param.rolloff)  # Avoid NaN
    elif pulse_type == "costails":
        nl = round(0.5 * (1 - tx_param.rolloff) * tx_param.duty * num_samp)  # start index of cos roll-off
        nr = tx_param.duty * num_samp - nl  # end index of cos roll-off

        num_mark = np.arange(0, nl)  # indices where the pulse is 1
        num_cos = np.arange(nl, nr)  # transition region of cos roll-off

        el_pulse[num_samp * num_sym / 2 + num_mark] = 1
        h_period = tx_param.duty * num_samp - 2 * nl
        if h_period != 0:
            el_pulse[num_cos + num_samp * num_sym / 2] = 0.5 * (
                    1 + np.cos(math.pi / h_period * (num_cos - nl + 0.5)))
        el_pulse[0: num_samp * num_sym / 2] = np.flipud(
            el_pulse[num_samp * num_sym / 2: num_samp * num_sym])  # first half of the pulse
    else:
        raise RuntimeError("The pulse ptype does not exist")
    return el_pulse
