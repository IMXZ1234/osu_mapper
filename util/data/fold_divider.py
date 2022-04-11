from sklearn import model_selection


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
