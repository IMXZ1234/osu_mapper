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
