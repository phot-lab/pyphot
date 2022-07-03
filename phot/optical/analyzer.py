import numpy as np
from phot.values import globals
import math
from .format import get_format_info
import matplotlib.pyplot as plt


class Analyzer:
    def __init__(self, mod_format: str):
        self._sym_rate = globals._sym_rate
        self.mod_format = mod_format

    def eval_eye(self, seq: np.ndarray, sig: np.ndarray, plot_eye: bool = False):
        if globals.SAMP_FREQ is not None:
            num_pt = int(globals.SAMP_FREQ / self._sym_rate)  # number of points per symbol
            if math.floor(num_pt) != num_pt:
                raise RuntimeError("Number of points per symbol is not an integer.")
            num_sym = globals.NUM_SAMP / num_pt
        else:
            raise RuntimeError("no eye at one samples per symbol")
        num_pol = np.shape(sig)[1]
        num_shift = round(num_pt / 2)  # the first bit is centered at index 1
        iric_mat = np.transpose(np.reshape(np.roll(sig, num_shift), (num_pt, int(num_sym * num_pol)), order='F'))
        iric_mat = np.real(iric_mat)  # Only need real part to draw eye

        format_info = get_format_info(self.mod_format)
        bot_vec = np.zeros((format_info.digit, num_pt))
        top_vec = np.zeros((format_info.digit, num_pt))

        for k in range(format_info.digit):
            mat_index = (seq == k).ravel()
            eye_sig = iric_mat[mat_index, :]
            top_vec[k, :] = np.min(eye_sig, 0)  # top of eye
            bot_vec[k, :] = np.max(eye_sig, 0)  # bottom of eye

        index_top = np.argsort(np.max(top_vec, 1))  # sort because of pattern
        index_bot = np.argsort(np.min(bot_vec, 1))

        # Eye opening: best of the worst among symbols
        eye_opening = np.max(top_vec[index_top[1:], :] - bot_vec[index_bot[:-1], :], 1)  # among samples
        eye_opening = np.max(eye_opening, 0)  # among alphabet

        if eye_opening < 0:
            eye_opening = math.nan

        # eye_opening = 10 * np.log10(eye_opening)  # convert [mw] to [dBm]

        if plot_eye:
            tim = np.arange(-1 / 2, 1 / 2, 1 / num_pt)
            plt.figure(figsize=(8, 6), layout='constrained')
            if num_pol == 1:
                plt.plot(tim, iric_mat.T, color=None)
            else:
                # dual polarization situation unfinished
                pass
            plt.xlabel('Normalized time [symbols]')
            plt.ylabel('Eye')
            plt.title('Eye Graph')
            plt.show()

        return eye_opening


def plot_constellation(sig: np.ndarray) -> None:
    axis_x = np.real(sig).ravel()
    axis_y = np.imag(sig).ravel()
    plt.scatter(axis_y, axis_x)
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.grid()
    plt.show()
