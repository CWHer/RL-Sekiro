import time
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def timeLog(func):
    @wraps(func)
    def clocked(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        print("{} finishes after {:.2f} s".format(
            func.__name__, time.time() - start))
        return ret
    return clocked


def plotLine(y: list, title: str):
    """[summary]

    Args:
        y (list): [description]. list of value to plot
    """
    _, ax = plt.subplots()
    ax.plot(range(0, len(y)), y)
    ax.set_title(title)
    plt.show()


def plotLines(data: list):
    """[summary]
    plot n lines

    Args:
        data (list): [description]. list of (y, title)
    """
    fig_num = len(data)
    _, figs = plt.subplots(1, fig_num)
    for i in range(fig_num):
        y, title = data[i]
        figs[i].plot(range(0, len(y)), y)
        figs[i].set_title(title)
    plt.show()


def plotSemilogy(y: list, title: str):
    """[summary]
    semi-log along y-axis

    """
    _, ax = plt.subplots()
    ax.semilogy(range(0, len(y)), y)
    ax.set_title(title)
    plt.show()


def plotHeatMap(p: np.ndarray, title: str):
    """[summary]

    Args:
        p (np.ndarray): [description]. 2-D matrix
    """
    fig, ax = plt.subplots()
    im = ax.imshow(p, cmap=cm.Reds)
    y_len, x_len = p.shape
    ax.set_xticks(np.arange(x_len))
    ax.set_yticks(np.arange(y_len))
    ax.set_title(title)
    fig.colorbar(im)
    plt.show()


def plotHeatMaps(states, titles):
    for state, title in zip(states, titles):
        plotHeatMap(state, title)


def plotSparseMatrix(p: np.ndarray, title: str):
    """[summary]
    use spy to visualize sparse matrix

    Args:
        p (np.ndarray): [description]. 2-D matrix
    """
    _, ax = plt.subplots()
    ax.spy(p)
    ax.set_title(title)
    plt.show()
