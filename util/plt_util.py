import os

import numpy as np
from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('QtAgg')


def plot_loss(loss_list, title=None, save_path=None, show=True):
    if len(loss_list) == 0:
        return
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(len(loss_list)) + 1, loss_list)
    if title is None:
        title = 'plot'
    ax.set_title(title)
    fig.canvas.draw()
    fig_array = np.array(fig.canvas.renderer.buffer_rgba())
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    return fig_array, fig


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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show:
        plt.show()
    return fig_array, fig


def plot_multiple_signal(signal_list, title=None, save_path=None, show=True, signal_len_per_inch=100):
    signal_len = len(signal_list[0])
    fig = plt.figure(figsize=[signal_len / signal_len_per_inch, 4.8])
    ax = fig.add_subplot()
    if title is None:
        title = 'plot'
    ax.set_title(title)

    fig_array_list = []
    for signal in signal_list:
        if len(signal) == 0:
            continue
        ax.plot(np.arange(len(signal)), signal)
        fig.canvas.draw()
        fig_array = np.array(fig.canvas.renderer.buffer_rgba())
        fig_array_list.append(fig_array)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.clf()
    return fig_array_list, fig


if __name__ == '__main__':
    signal = np.arange(15)
    fig_array, fig = plot_signal(signal, show=True)
    print(fig_array)
    plt.imshow(fig_array)
    plt.show()
