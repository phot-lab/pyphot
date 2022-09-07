"""
Copyright (c) 2022 Beijing Jiaotong University
PhotLab is licensed under [Open Source License].
You can use this software according to the terms and conditions of the [Open Source License].
You may obtain a copy of [Open Source License] at: [https://open.source.license/]

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.

See the [Open Source License] for more details.

Author: Chunyu Li
Created: 2022/9/6
Supported by: National Key Research and Development Program of China
"""

import numpy as np
import matplotlib.pyplot as plt

_plot = True


def config(plot=True):
    global _plot
    _plot = plot


def plot_scatter(sig: np.ndarray, pt_size=5) -> None:
    if not _plot:
        return
    axis_x = np.real(sig).ravel()
    axis_y = np.imag(sig).ravel()
    # plt.figure(figsize=(10, 10), layout='constrained')
    plt.figure(figsize=(10, 10))
    plt.scatter(axis_y, axis_x, s=pt_size)
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.title('Scatter Plot')
    plt.grid()
    plt.show()


def plot_linechart(x, y, xlabel=None, ylabel=None, title=None):
    if not _plot:
        return
    # plt.figure(figsize=(10, 10), layout='constrained')
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, color=None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
