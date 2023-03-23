from ..utils import plot_scatter

import matplotlib.pyplot as plt
import numpy as np


def constellation_diagram(signals):
    plot_scatter(signals[0], pt_size=1)
