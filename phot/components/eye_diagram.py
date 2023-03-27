import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from .. import settings
from numba import njit


def eye_diagram(signals, up_sampling_factor):
    if not settings._plot:
        return

    plt.figure(figsize=(10, 10))

    _eye_diagram(np.real(signals[0][:8000].ravel()), up_sampling_factor * 2, cmap=plt.cm.coolwarm)

    plt.ylim(-5, 5)
    plt.xlim(-8, 8)
    plt.title("Eye diagram")
    plt.xlabel("Time [ps]")  # ps = 皮秒
    plt.ylabel("Power Level [a.u.]")  # a.u. = arbitrary units
    plt.show()


def _eye_diagram(y, window_size, offset=0, colorbar=True, **imshowkwargs):
    """
    Plot an eye diagram using matplotlib by creating an image and calling
    the `imshow` function.
    <common>
    """
    counts = grid_count(y, window_size, offset)
    counts = counts.astype(np.float32)
    counts[counts == 0] = np.nan
    ymax = y.max()
    ymin = y.min()
    yamp = ymax - ymin
    min_y = ymin - 0.05 * yamp
    max_y = ymax + 0.05 * yamp
    x = counts.T[::-1, :]  # array-like or PIL image
    extent = [-8, 8, min_y, max_y]  # floats (left, right, bottom, top)
    plt.imshow(X=x, extent=extent, **imshowkwargs)
    ax = plt.gca()
    ax.set_facecolor("k")
    plt.grid(color="w")
    if colorbar:
        plt.colorbar()


def grid_count(y, window_size, offset=0, size=None, fuzz=True, bounds=None):
    """
    Parameters
    ----------
    `y` is the 1-d array of signal samples.

    `window_size` is the number of samples to show horizontally in the
    eye diagram.  Typically this is twice the number of samples in a
    "symbol" (i.e. in a data bit).

    `offset` is the number of initial samples to skip before computing
    the eye diagram.  This allows the overall phase of the diagram to
    be adjusted.

    `size` must be a tuple of two integers.  It sets the size of the
    array of counts, (height, width).  The default is (800, 640).

    `fuzz`: If True, the values in `y` are reinterpolated with a
    random "fuzz factor" before plotting in the eye diagram.  This
    reduces an aliasing-like effect that arises with the use of
    Bresenham's algorithm.

    `bounds` must be a tuple of two floating point values, (ymin, ymax).
    These set the y range of the returned array.  If not given, the
    bounds are `(y.min() - 0.05*A, y.max() + 0.05*A)`, where `A` is
    `y.max() - y.min()`.

    Return Value
    ------------
    Returns a numpy array of integers.

    """
    if size is None:
        size = (800, 640)
    height, width = size
    dt = width / window_size
    counts = np.zeros((width, height), dtype=np.int32)

    if bounds is None:
        ymin = y.min()
        ymax = y.max()
        yamp = ymax - ymin
        ymin = ymin - 0.05 * yamp
        ymax = ymax + 0.05 * yamp
    else:
        ymin, ymax = bounds

    start = offset
    while start + window_size < len(y):
        end = start + window_size
        yy = y[start : end + 1]
        k = np.arange(len(yy))
        xx = dt * k
        if fuzz:
            f = interp1d(xx, yy, kind="cubic")
            jiggle = dt * (np.random.beta(a=3, b=3, size=len(xx) - 2) - 0.5)
            xx[1:-1] += jiggle
            yd = f(xx)
        else:
            yd = yy
        iyd = (height * (yd - ymin) / (ymax - ymin)).astype(np.int32)
        bres_curve_count(xx.astype(np.int32), iyd, counts)

        start = end
    return counts


@njit
def bres_segment_count(x0, y0, x1, y1, grid, endpoint):
    """Bresenham's algorithm.

    See http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """

    # cdef unsigned nrows, ncols
    # cdef int e2, sx, sy, err
    # cdef int dx, dy

    nrows = grid.shape[0]
    ncols = grid.shape[1]

    if x1 > x0:
        dx = x1 - x0
    else:
        dx = x0 - x1
    if y1 > y0:
        dy = y1 - y0
    else:
        dy = y0 - y1

    sx = 0
    if x0 < x1:
        sx = 1
    else:
        sx = -1
    sy = 0
    if y0 < y1:
        sy = 1
    else:
        sy = -1

    err = dx - dy

    while True:
        # When endpoint is 0, this test occurs before we increment the
        # grid value, so we don't count the last point.
        if endpoint == 0 and x0 == x1 and y0 == y1:
            break

        if (0 <= x0 < nrows) and (0 <= y0 < ncols):
            grid[x0, y0] += 1

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return 0


def bres_curve_count(x, y, grid):

    for k in range(len(x) - 1):
        x0 = x[k]
        y0 = y[k]
        x1 = x[k + 1]
        y1 = y[k + 1]
        bres_segment_count(x0, y0, x1, y1, grid, 0)

    if 0 <= x1 < grid.shape[0] and 0 <= y1 < grid.shape[1]:
        # Count the last point in the curve.
        grid[x1, y1] += 1
