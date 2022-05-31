from preprocess.dataset.fit_dataset import FitDataset


class DownSampler:
    def __init__(self, from_ds: FitDataset, to_ds: FitDataset):
        self.from_ds = from_ds
        self.to_sd = to_ds

    def down_sample_raw(self):
        from_label = self.from_ds.get_raw_label()
        # keep_idx