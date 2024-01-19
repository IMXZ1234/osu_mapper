import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import yaml

from system.base_sys import Train
from system.gan_sys import TrainGAN, TrainACGANWithinBatch
from system.rnngan_sys import TrainRNNGANPretrain, TrainSeqGANAdvLoss
from system.seq2seq_sys import TrainSeq2Seq
from system.vae_sys import TrainVAE
from system.wgan_sys import TrainWGAN, TrainWGANWithinBatch, TrainACWGANWithinBatch
from system.word2vec_skipgram_sys import TrainWord2VecSkipGram
from system.regression_sys import TrainRegression
from system.cddpm_sys import CDDPMSYS
from util.general_util import try_format_dict_with_path

np.set_printoptions(suppress=True)
SYS_DICT = {
    'basic': Train,
    'regression': TrainRegression,
    'rnngan_with_pretrain': TrainRNNGANPretrain,
    'gan': TrainGAN,
    'acganwb': TrainACGANWithinBatch,
    'wgan': TrainWGAN,
    'wganwb': TrainWGANWithinBatch,
    'acwganwb': TrainACWGANWithinBatch,
    'vae': TrainVAE,
    'seq2seq': TrainSeq2Seq,
    'seqgan_adv_loss': TrainSeqGANAdvLoss,
    'skipgram': TrainWord2VecSkipGram,
    'cddpm': CDDPMSYS,
}
print('init sysdict')


def plot_difference_distribution(pred_list, target_list, difference_step):
    difference_list = []
    for i in range(len(pred_list)):
        difference_list.append(pred_list[i] - target_list[i])
    max_pos_difference = np.max(difference_list)
    max_neg_difference = np.min(difference_list)
    section_interval = (max_pos_difference - max_neg_difference) / difference_step
    y = np.zeros(difference_step)
    for i in range(len(difference_list)):
        section_index = math.floor((difference_list[i] - max_neg_difference) / section_interval)
        if section_index == difference_step:
            section_index -= 1
        y[section_index] += 1
    x = list(range(difference_step))
    for i in range(difference_step):
        x[i] = x[i] * section_interval + max_neg_difference + section_interval / 2
    plt.plot(x, y)
    plt.show()


def train_with_config(config_path, format_config=False, folds=5):
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    if folds is None:
        print('train with no cross_val')
        sys = SYS_DICT[config_dict.get('train_type', 'basic')](config_dict)
        sys.run_train()
    else:
        for fold in range(1, folds + 1):
            print('Fold %d' % fold)
            if format_config:
                formatted_config_dict = get_fold_config(config_dict, fold)
            else:
                formatted_config_dict = config_dict
            sys = SYS_DICT[formatted_config_dict.get('train_type', 'basic')](formatted_config_dict)
            sys.run_train()


def get_fold_config(config_dict, fold):
    fold_config_dict = copy.deepcopy(config_dict)
    # possible items which needs formatting with fold
    for path in [
        ['data_arg', 'test_dataset_arg', 'table_name'],
        ['data_arg', 'test_dataset_arg', 'data_path'],
        ['data_arg', 'test_dataset_arg', 'label_path'],
        ['data_arg', 'train_dataset_arg', 'table_name'],
        ['data_arg', 'train_dataset_arg', 'data_path'],
        ['data_arg', 'train_dataset_arg', 'label_path'],
        ['output_arg', 'log_dir'],
        ['output_arg', 'model_save_dir'],
    ]:
        try_format_dict_with_path(fold_config_dict, path, fold)
    # print(fold_config_dict)
    return fold_config_dict
