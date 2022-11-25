import numpy as np
import math
from deprecated.phot.values import globals
from .signal import my_filter
from .sample import seq2samp
from .signal import up_sample
from deprecated.phot.utils import mathutils
from deprecated.phot.utils import logger


class RxDsp:
    def __init__(self, elec_filter_type: str,
                 elec_bandwidth,
                 recovery: str,
                 mod_format: str,
                 seq=None,
                 interp: str = 'fine',
                 elec_param=math.nan
                 ):
        if globals.SAMP_FREQ is not None:
            num_pts = globals.SAMP_FREQ / globals._sym_rate  # number of points per symbol
            if np.floor(num_pts) != num_pts:
                raise RuntimeError('Number of points per symbol is not an integer.')
            pass
        else:
            num_pts = 1
        self.elec_filter_type = elec_filter_type
        self.elec_bandwidth = elec_bandwidth
        self.elec_param = elec_param
        self._num_pts = num_pts
        self.recovery = recovery
        self.mod_format = mod_format
        self.interp = interp
        self.seq = seq

    def process(self, elec_sig):
        """
        Process digital signal to get complex symbols

        Args:
            elec_sig: digital signal

        Returns:
            complex symbols
        """

        # 1. apply the post-detection electrical filter (ADC filter)
        elec_sig = lowpass_filter(elec_sig, self.elec_filter_type, self.elec_bandwidth, self.elec_param)

        # 2. recover time delay
        elec_sig = clock_recovery(elec_sig, self.seq, self.recovery, self.interp, self.mod_format)

        # 3. Sample (or downsample and MIMO)
        ak = elec_sig[0::int(self._num_pts), :]

        # 4. Automatic gain control
        symbols, gain_factor = agc(self.seq, ak, self.mod_format)

        return symbols, gain_factor


def lowpass_filter(elec_sig, elec_filter_type: str, elec_bandwidth, elec_param=math.nan):
    """
    Args:
        elec_sig:
        elec_filter_type: elec_filter_type: electrical filter (LPF) type (see MYFILTER). Such a
            filter is an antialiasing filter before ideal sampling.
        elec_bandwidth: LPF bandwidth normalized to SYMBRATE.
        elec_param: electrical filter extra parameters

    Returns:
        Electrical signal after filtered
    """

    if globals.FN is not None:
        factor_norm = globals.FN / globals._sym_rate
    else:
        num_sym = np.shape(elec_sig)[0]
        factor_norm = np.arange(-1 / 2, 1 / 2, 1 / num_sym)
    hf = my_filter(elec_filter_type, factor_norm, elec_bandwidth, elec_param)  # lowpass filter
    for k in range(np.shape(elec_sig)[1]):
        temp = np.fft.fft(elec_sig[:, k], axis=0)
        elec_sig[:, k] = np.fft.ifft(temp * hf, axis=0)
    return elec_sig


def clock_recovery(signal: np.ndarray, seq: np.ndarray, recovery: str, interp: str, mod_format: str):
    num_cols = np.shape(signal)[1]  # number of signals
    if num_cols != np.shape(seq)[1]:
        raise RuntimeError('The sequence must be in decimal form.')
    rho = np.zeros((num_cols, num_cols))
    ns = np.zeros((num_cols, num_cols))
    for m1 in range(num_cols):
        for m2 in range(num_cols):
            ns[m1][m2], rho[m1][m2] = time_estimate(np.reshape(signal[:, m1], (-1, 1)), np.reshape(seq[:, m2], (-1, 1)),
                                                    recovery, interp, mod_format)
    index_max = np.argmax(rho, axis=1)  # best candidate. imax(m1)~= m1 is an indication of signal exchange.
    if len(np.unique(index_max)) != len(index_max):
        logger.warning('Clock recovery by the function clockrec failed.')
    else:
        signal = signal[:, index_max]  # signal exchange (if necessary)
        for m1 in range(num_cols):
            fix_delay(np.reshape(signal[:, m1], (-1, 1)), ns[m1, index_max[m1]])
    return signal


def time_estimate(signal: np.ndarray, seq, recovery: str, interp: str, mod_format: str):
    if recovery is None:
        raise RuntimeError('Unknown timing recovery method.')

    # Timing recovery
    if recovery == 'da':
        ns, rho = mlda_time(signal, seq, interp, mod_format)
    elif recovery == '':
        ns = math.nan
        rho = math.nan
    else:
        raise RuntimeError('Unknown timing recovery method.')

    return ns, rho


def mlda_time(signal: np.ndarray, seq, interp: str, mod_format: str):
    # Init
    size_fft = len(signal)
    num_sym = len(seq)
    num_pts = size_fft / num_sym
    if np.remainder(num_pts, 1) != 0:
        raise RuntimeError('Fractional number of samples per symbol has not yet done.')

    # Go
    ak_id = seq2samp(seq, mod_format)
    aku = up_sample(ak_id, num_pts)
    aku = np.fft.fft(aku, axis=0)
    corr = np.abs(np.fft.ifft(np.conj(aku) * np.fft.fft(signal, axis=0), axis=0))  # correlation
    rho = np.max(corr)
    index_max = np.argmax(corr)

    # Fine tuning
    if interp is not None:
        if interp == 'fine':  # fine tuning
            i1 = mathutils.n_mod(index_max, size_fft) - 1
            i2 = mathutils.n_mod(index_max + 1, size_fft) - 1
            i3 = mathutils.n_mod(index_max + 2, size_fft) - 1
            # max of parabola passing for x-coordinates [1 2 3] with values corr.
            ifine = (5 * corr[i1] + 3 * corr[i3] - 8 * corr[i2]) / (corr[i1] + corr[i3] - 2 * corr[i2]) / 2
            index_max = mathutils.n_mod(ifine + index_max - 2, size_fft)  # -2: because central point is at coordinate 2
        elif interp == 'nearest':  # coarse (but faster) tuning
            # nothing to do
            pass
        else:
            raise RuntimeError('Unknown time interpolation method.')
    return index_max, rho


def fix_delay(signal: np.ndarray, ns):
    # apply delay recovery

    if np.isscalar(ns):
        if ns.is_integer():
            signal = np.roll(signal, -ns)
        else:
            omega = 2 * math.pi * globals.FN / globals.SAMP_FREQ
            omega = np.reshape(omega, (-1, 1))
            signal = np.fft.ifft(np.fft.fft(signal, axis=0) * mathutils.fast_exp(omega * ns), axis=0)
    else:
        for k in range(len(ns)):
            if np.floor(ns[k]) == ns[k]:
                signal[:, k] = np.roll(signal[:, k], -ns[k])
            else:
                omega = 2 * math.pi * globals.FN / globals.SAMP_FREQ
                omega = np.reshape(omega, (-1, 1))
                signal[:, k] = np.fft.ifft(np.fft.fft(signal[:, k], axis=0) * mathutils.fast_exp(omega * ns[k]), axis=0)

    return signal


def agc(seq: np.ndarray, ak: np.ndarray, mod_format: str):
    # AGC automatic gain control
    num_pols = np.shape(ak)[1]  # number of polarizations
    num_bits = np.shape(seq)[1] / num_pols  # number of bits per polarization

    if num_pols == 1:
        data_id = seq2samp(seq, mod_format)
        gain_factor = np.sum(ak * np.conj(data_id)) / np.sum(data_id * np.conj(data_id))
        ak = ak / gain_factor
    else:
        raise RuntimeError('Not complete yet!')
    return ak, gain_factor
