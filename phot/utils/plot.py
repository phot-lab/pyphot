import numpy as np
import matplotlib.pyplot as plt


def plot_scatter(sig: np.ndarray, pt_size=5) -> None:
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
    # plt.figure(figsize=(10, 10), layout='constrained')
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, color=None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
