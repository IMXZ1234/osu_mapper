import os
import pickle
import logging
import numpy as np
from sklearn import model_selection


def _save_items(item_paths, items):
    for item_path, item in zip(item_paths, items):
        dirname = os.path.dirname(item_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(item_path, 'wb') as f:
            pickle.dump(item, f)


def _load_items(item_paths):
    items = []
    for item_path in item_paths:
        try:
            with open(item_path, 'rb') as f:
                items.append(pickle.load(f))
        except Exception:
            items.append(None)
            print('fail to load %s' % item_path)
            continue
    return items


def fold_item_names(item_names, fold, phase='train'):
    return [
        '_'.join([phase + str(fold), item_name])
        for item_name in item_names
    ]


def item_names_to_paths(save_dir, item_names):
    return [
        os.path.join(save_dir, item_name + '.pkl')
        for item_name in item_names
    ]

def raw_item_names_to_fold_item_names(item_names, folds, phases):
    return {
        phase: {
            fold: fold_item_names(item_names, fold, phase)
            for fold in range(1, folds + 1)
        } for phase in phases
    }


class FitDataset:
    DEFAULT_ITEM_NAMES = ['data', 'label']
    PHASES = ['train', 'test']

    def __init__(self, save_dir, item_names=DEFAULT_ITEM_NAMES, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.Logger(self.__class__.__name__)
        self.save_dir = save_dir

        # raw items
        self.items = None
        self.raw_save_dir = os.path.join(self.save_dir, 'raw')
        # raw item names
        self.item_names = item_names
        self.raw_item_paths = item_names_to_paths(self.save_dir, self.item_names)

        self.folds = None
        self.fold_items = None
        self.fold_save_dir = os.path.join(save_dir, 'fold')
        self.fold_item_names = None
        self.fold_item_paths = None

        self.data_idx = item_names.index('data')
        self.label_idx = item_names.index('label')

    def get_raw_items(self):
        self._try_load_raw()
        return self.items

    def get_raw_data(self):
        return self.get_raw_items()[self.data_idx]

    def get_raw_label(self):
        return self.get_raw_items()[self.label_idx]

    def save_raw(self):
        _save_items(self.raw_item_paths, self.items)

    def load_raw(self, warning=True):
        self.items = _load_items(self.raw_item_paths)
        for i, item in enumerate(self.items):
            if warning and item is None:
                self.logger.warning('fail to load %s' % self.raw_item_paths[i])

    def get_raw_item(self, item_name):
        if item_name not in self.item_names:
            print('%s does not exist' % item_name)
            return None
        return self.items[self.item_names.index(item_name)]

    def _try_load_raw(self, warning=True):
        if self.items is None:
            if not self.raw_exists():
                self.logger.error('raw data does not exist!')
            self.load_raw(warning)

    def raw_exists(self):
        for item_path in self.raw_item_paths:
            if not os.path.exists(item_path):
                self.logger.warning('%s does not exist!' % item_path)
                return False
        return True

    def get_fold_items(self, fold, phase='train'):
        self._try_load_fold(fold)
        return self.fold_items[phase][fold]

    def get_fold_item(self, item_name, fold, phase='train'):
        if item_name not in self.item_names:
            print('%s does not exist' % item_name)
            return None
        return self.get_fold_items(fold, phase)[self.item_names.index(item_name)]

    def get_fold_data(self, fold, phase='train'):
        return self.get_fold_items(fold, phase)[self.data_idx]

    def get_fold_label(self, fold, phase='train'):
        return self.get_fold_items(fold, phase)[self.label_idx]

    def _try_load_fold(self, fold, warning=True):
        should_load = False
        if self.fold_items is None:
            should_load = True
        else:
            for phase in self.PHASES:
                if phase not in self.fold_items:
                    should_load = True
                    break
                if fold not in self.fold_items[phase]:
                    should_load = True
                    break
        if should_load:
            if not self.fold_exists(fold):
                self.logger.error('fold %d data does not exist!' % fold)
            self.load_fold(fold, warning)

    def fold_exists(self, folds=5):
        if self.fold_item_paths is None:
            self.folds = folds
            self.init_fold_item_names_paths()
        for phase_dict in self.fold_item_paths.values():
            for fold in range(1, 1+folds):
                for item_path in phase_dict[fold]:
                    if not os.path.exists(item_path):
                        self.logger.warning('%s does not exist!' % item_path)
                        return False
        return True

    def save_fold(self, fold, remove_from_mem=True):
        for phase in self.PHASES:
            _save_items(self.fold_item_paths[phase][fold], self.fold_items[phase][fold])
            if remove_from_mem:
                del self.fold_items[phase][fold]

    def load_fold(self, fold, warning=True):
        if self.fold_items is None:
            self.fold_items = {
                phase: {} for phase in self.PHASES
            }
        for phase in self.PHASES:
            self.fold_items[phase][fold] = _load_items(self.fold_item_paths[phase][fold])

    def init_fold_item_names_paths(self):
        self.fold_item_names = raw_item_names_to_fold_item_names(
            self.item_names, self.folds, self.PHASES
        )
        self.fold_item_paths = {
            phase: {
                fold: item_names_to_paths(self.save_dir, item_names)
                for fold, item_names in phase_dict.items()
            } for phase, phase_dict in self.fold_item_names.items()
        }

    def div_folds(self, folds=5, save_first=None, shuffle=False):
        self.folds = folds
        if save_first is None:
            save_first = self.folds
        self._try_load_raw()
        kf = model_selection.KFold(self.folds, shuffle=shuffle)
        self.fold_items = {
            phase: {} for phase in self.PHASES
        }
        self.init_fold_item_names_paths()
        sample_num = len(self.items[0])
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(list(range(sample_num)))):
            if save_first <= 0:
                break
            save_first -= 1
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)
            print('train_idx')
            print(train_idx)
            print('test_idx')
            print(test_idx)
            # print('self.items[1]')
            # print(self.items[1].shape)
            # print('self.items[0]')
            # print(self.items[0].shape)
            if isinstance(self.items[0], np.ndarray):
                fold_train_items = [
                    item[train_idx]
                    for item in self.items
                ]
                fold_test_items = [
                    item[test_idx]
                    for item in self.items
                ]
            else:
                # list
                fold_train_items = [
                    [item[idx] for idx in train_idx]
                    for item in self.items
                ]
                fold_test_items = [
                    [item[idx] for idx in test_idx]
                    for item in self.items
                ]
            self.fold_items['train'][fold_idx+1] = fold_train_items
            self.fold_items['test'][fold_idx+1] = fold_test_items
            self.save_fold(fold=fold_idx + 1)

    def get_raw_item_with_idx(self, idx_list):
        return [
            [item[idx] for idx in idx_list]
            if isinstance(item, (list, tuple))
            else item[np.asarray(idx_list, dtype=int)]
            for item in self.get_raw_items()
        ]
