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


def divisor_distribution(divisor_list):
    possible_divisors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16]
    divisor_count = [0 for _ in range(len(possible_divisors))]
    back = [None for _ in range(max(possible_divisors) + 1)]
    for i, divisor in enumerate(possible_divisors):
        back[divisor] = i
    for divisor in divisor_list:
        divisor_count[back[divisor]] += 1
    return divisor_count
