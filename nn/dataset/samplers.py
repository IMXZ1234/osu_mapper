import math

from torch.utils.data import Sampler
import random


class BucketBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, rand_seed=None):
        super().__init__(data_source)
        self.rand_inst = random.Random(rand_seed)
        self.data_source = data_source
        self.batch_size = batch_size
        self.buckets_idx_list = []
        self.batch_bucket = []

        for bucket_id, (si, ei) in enumerate(self.data_source.bucket_to_range):
            self.buckets_idx_list.append(
                list(range(si, ei))
            )
            self.batch_bucket.extend([bucket_id] * math.ceil((ei - si) / self.batch_size))

    def __iter__(self):
        batch_bucket = self.batch_bucket.copy()
        buckets_idx_list = [idx_set.copy() for idx_set in self.buckets_idx_list]
        self.rand_inst.shuffle(batch_bucket)
        for bucket_idx in batch_bucket:
            this_batch = []
            idx_list = buckets_idx_list[bucket_idx]
            for _ in range(min(len(idx_list), self.batch_size)):
                chosen = self.rand_inst.randint(0, len(idx_list)-1)
                this_batch.append(idx_list[chosen])
                idx_list.pop(chosen)
            # print('sampler', this_batch)
            yield this_batch

    def __len__(self):
        return len(self.batch_bucket)
