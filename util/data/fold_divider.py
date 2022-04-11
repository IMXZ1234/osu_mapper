import os

from sklearn import model_selection


class AudioOsuDataFoldDivider:
    def __init__(self, data):
        self.data = data

    def div_folds(self, fold_dir, folds=5, shuffle=False):
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        kf = model_selection.KFold(folds, shuffle=shuffle)
        fold = 1
        for train_index, test_index in kf.split(self.data.audio_osu_list):
            print('generating folds %d data...' % fold)
            self.data.from_index(self.data, train_index, os.path.join(fold_dir, 'train%d.pkl' % fold)).save()
            self.data.from_index(self.data, test_index, os.path.join(fold_dir, 'test%d.pkl' % fold)).save()
            fold += 1


class OsuTrainDBFoldDivider:
    def __init__(self, folds=5, shuffle=False):
        self.folds = folds
        self.shuffle = shuffle

    def div_folds(self, sample_num):
        """
        Returns a generator giving out train_idx, test_idx
        """
        kf = model_selection.KFold(self.folds, shuffle=self.shuffle)
        return kf.split(list(range(sample_num)))


class OsuTrainDBDebugFoldDivider:
    def __init__(self, train_fold_sample_num=1, test_fold_sample_num=1, folds=5, shuffle=False):
        self.train_fold_sample_num = train_fold_sample_num
        self.test_fold_sample_num = test_fold_sample_num
        self.folds = folds

    def div_folds(self, sample_num):
        """
        Returns a generator giving out train_idx, test_idx
        """
        assert sample_num >= self.train_fold_sample_num + self.test_fold_sample_num
        return (([i for i in range(self.train_fold_sample_num)],
                [i + self.train_fold_sample_num for i in range(self.test_fold_sample_num)])
                for _ in range(self.folds))
