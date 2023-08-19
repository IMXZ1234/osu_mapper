import numpy as np
from matplotlib import pyplot as plt


def plot_loss(loss_list, title=None, save_path=None, show=True):
    if len(loss_list) == 0:
        return
    plt.figure()
    plt.plot(np.arange(len(loss_list)) + 1, loss_list)
    if title is None:
        title = 'plot'
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.clf()


def plot_signal(signal, title=None, save_path=None, show=True):
    if len(signal) == 0:
        return
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(len(signal)), signal)
    if title is None:
        title = 'plot'
    ax.set_title(title)
    fig.canvas.draw()
    fig_array = np.array(fig.canvas.renderer.buffer_rgba())
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.clf()
    return fig_array
