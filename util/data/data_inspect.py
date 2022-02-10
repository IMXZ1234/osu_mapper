import pickle

import slider
from matplotlib import pyplot as plt
import numpy as np


class DataInspector:
    def __init__(self, data):
        self.data = data

    def attr_distribution(self, attr, bins=None, display=True):
        attr_list = self.data.get_attr_list(attr)
        if bins is None:
            bins = np.linspace(np.min(attr_list), np.max(attr_list), 25)
        if display:
            plt.hist(attr_list, bins)
            plt.show()
        return attr_list

