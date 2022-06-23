import math
import os
import pickle
import random

import numpy as np
import slider

from nn.dataset import dataset_util
from preprocess import db, prepare_data
from preprocess.preprocessor import MelPreprocessor
from util import beatmap_util, general_util
from preprocess.dataset import fit_dataset


class RNNDataset(fit_dataset.FitDataset):
    DEFAULT_SAVE_DIR = r'./resources/cond_data/fit/rnn/'
    """
    A snap may be of label 0-5:
    We predict label of every snap with features:
    [
        one-hot label of classes of label_snap former snaps,
        mel features of audio_mel around target snap,
        # following two features are now appended after each sample's features
        speed_star,
        bpm,
    ]
    """
    def __init__(self, save_dir,
                 db_path=prepare_data.DEFAULT_TRAIN_MEL_DB_PATH,
                 audio_dir=prepare_data.DEFAULT_TRAIN_MEL_AUDIO_DIR,
                 audio_mel=4, snap_mel=4, snap_divisor=8,
                 take_first=100, random_seed=None,
                 coeff_speed_stars=2.5,
                 coeff_bpm=120,
                 label_num=3,
                 item_names=fit_dataset.FitDataset.DEFAULT_ITEM_NAMES, logger=None,
                 preprocess_arg=dict(),
                 **kwargs):
        """
        subseq_beat: length of subsequences.
        audio_mel: number of mel frames around target snap to be used as feature.
        """
        super().__init__(save_dir, item_names, logger)
        self.db = db.OsuDB(db_path)
        self.audio_dir = audio_dir

        self.take_first = take_first
        self.random_seed = random_seed

        self.audio_mel = audio_mel
        self.snap_mel = snap_mel
        self.snap_divisor = snap_divisor
        self.half_audio_mel = audio_mel // 2
        self.half_audio_snap = math.ceil(self.half_audio_mel / self.snap_mel)

        self.coeff_speed_stars = coeff_speed_stars
        self.coeff_bpm = coeff_bpm

        self.label_num = label_num
        self.eye = np.eye(self.label_num)
        self.preprocess_arg = preprocess_arg

    def make_feature(self, mel, label, speed_stars, bpm):
        """
        label of former snaps to one hot, and concatenate all features of this snap(sample) into a vector
        """
        label = np.asarray(label, dtype=int)
        one_hot_feature = self.eye[label].reshape([-1])
        # print('mel.shape')
        # print(mel.shape)
        mel = mel.reshape([-1])
        return np.concatenate([mel,
                               one_hot_feature,
                               np.array([speed_stars / self.coeff_speed_stars]),
                               np.array([bpm / self.coeff_bpm]),
                               ])

    def preprocess_audio(self, audio_path, beatmap):
        # to 1 channel
        preprocessor = MelPreprocessor(**self.preprocess_arg)
        path_to = r'./resources/cond_data/temp/%s' % general_util.change_ext(os.path.basename(audio_path), 'pkl')
        path_to_snap_offset = r'./resources/cond_data/temp/snap_offset_%s' % general_util.change_ext(os.path.basename(audio_path), 'pkl')
        print('path_to')
        print(path_to)
        if not os.path.exists(path_to):
            # pos for snap offset
            snap_offset = preprocessor.preprocess(beatmap, audio_path, path_to)[1]
            with open(path_to_snap_offset, 'wb') as f:
                pickle.dump(snap_offset, f)
        else:
            with open(path_to_snap_offset, 'rb') as f:
                snap_offset = pickle.load(f)
        with open(path_to, 'rb') as f:
            audio_data = pickle.load(f).numpy()
        print(audio_data.shape)
        return audio_data, snap_offset

    def trim_audio_data(self, audio_data, snap_offset):
        audio_data = np.mean(audio_data, axis=0)
        start_snap = max(snap_offset, self.half_audio_snap)
        start_mel = start_snap * self.snap_mel - self.half_audio_mel

        end_snap = (audio_data.shape[1] - start_snap * self.snap_mel - self.half_audio_mel) // self.snap_mel + start_snap
        end_mel = end_snap * self.snap_mel + self.half_audio_mel
        audio_data = audio_data / np.max(audio_data)
        audio_data, warmup_audio_data = audio_data[:, start_mel:end_mel], audio_data[:, :start_mel]
        return audio_data, warmup_audio_data

    class RNNSampleDataGenerator:
        def __init__(self, dataset, audio_path, beatmap):
            self.dataset = dataset
            audio_data, snap_offset = dataset.preprocess_audio(audio_path, beatmap)
            self.audio_data, self.warmup_audio_data = dataset.trim_audio_data(audio_data, snap_offset)
            self.beatmap = beatmap
            self.sample_num = self.audio_data.shape[1] // self.dataset.audio_mel
            self.warmup_sample_num = self.warmup_audio_data.shape[1] // self.dataset.audio_mel
            self.status = [0, self.sample_num, self.audio_data]
            self.warmup_status = [0, self.warmup_sample_num, self.warmup_audio_data]

        def __len__(self):
            return self.sample_num

        def warmup_len(self):
            return self.warmup_sample_num

        def get_next_sample_feature(self, last_label, warmup=False):
            if warmup:
                status = self.warmup_status
            else:
                status = self.status
            if status[0] >= status[1]:
                return None
            start_mel = status[0] * self.dataset.audio_mel
            feature = self.dataset.make_feature(
                status[2][:, start_mel:start_mel+self.dataset.audio_mel],
                last_label,
                self.beatmap.speed_stars(),
                self.beatmap.bpm_min(),
            ).reshape([1, -1])
            # print('feature.shape')
            # print(feature.shape)
            status[0] = status[0] + 1
            # print(feature)
            return feature

    def get_sample_data_generator(self, audio_path, beatmap):
        return self.RNNSampleDataGenerator(self, audio_path, beatmap)

    def get_db_sample_data_generator(self, db_path, audio_dir):
        return self.RNNDBSampleDataGenerator(self, db_path, audio_dir)

    class RNNDBSampleDataGenerator(RNNSampleDataGenerator):
        def __init__(self, dataset, db_path, audio_dir):
            self.dataset = dataset
            self.db = db.OsuDB(db_path)

            record = self.db.get_record(0, 'FILTERED')
            first_ho_snap = record[db.OsuDB.EXTRA_START_POS + 1]
            beatmap = pickle.loads(record[db.OsuDB.BEATMAP_POS])
            audio_path = os.path.join(audio_dir, record[db.OsuDB.AUDIOFILENAME_POS])
            with open(audio_path, 'rb') as f:
                audio_data = pickle.load(f)
            audio_data = audio_data.numpy()

            self.audio_data, self.audio_label = dataset.prepare_record(audio_data, beatmap, first_ho_snap)
            # skip the last feature which is last label(true label)
            # self.audio_data = self.audio_data[:, :-(self.dataset.label_num+2)]
            self.warmup_audio_data = None
            self.beatmap = beatmap
            self.sample_num = self.audio_data.shape[0]
            self.warmup_sample_num = 0
            self.status = [0, self.sample_num, self.audio_data]
            self.warmup_status = [0, self.warmup_sample_num, self.warmup_audio_data]

        def get_audio_label(self):
            return self.audio_label

        def get_next_sample_feature(self, last_label, warmup=False):
            if warmup:
                status = self.warmup_status
            else:
                status = self.status
            if status[0] >= status[1]:
                return None
            # feature = self.dataset.make_feature(
            #     status[2][status[0]],
            #     last_label,
            #     self.beatmap.speed_stars(),
            #     self.beatmap.bpm_min(),
            # ).reshape([1, -1])
            feature = status[2][status[0]].reshape([1, -1])
            # print('feature.shape')
            # print(feature.shape)
            status[0] = status[0] + 1
            # print(feature)
            return feature

    def prepare_record(self, audio_data, beatmap, first_ho_snap):
        """
        prepare sample cond_data from records in db
        sample cond_data will include last true label as feature
        """
        # to 1 channel
        audio_data = np.mean(audio_data, axis=0)
        # print('crop_start_time %d' % crop_start_time)
        try:
            speed_stars = beatmap.speed_stars()
        except Exception:
            return None, None
        assert isinstance(beatmap, slider.Beatmap)
        snap_ms = beatmap_util.get_snap_milliseconds(beatmap)
        total_snaps = beatmap_util.get_total_snaps(beatmap, self.snap_divisor)
        first_ho_time = beatmap_util.get_first_hit_object_time_milliseconds(beatmap)
        # one label per snap
        # three classes by default
        beatmap_label = dataset_util.hitobjects_to_label_v2(
            beatmap,
            first_ho_time,
            snap_ms,
            total_snaps,
            None,
            multi_label=True
        )
        start_snap = max(first_ho_snap, self.half_audio_snap)
        end_snap = min(
            # snaps_from_start_label
            len(beatmap_label) + first_ho_snap,
            # available snaps in audio cond_data - self.half_audio_mel
            (audio_data.shape[1] - start_snap * self.snap_mel - self.half_audio_mel) // self.snap_mel + start_snap
        )
        start_mel = start_snap * self.snap_mel - self.half_audio_mel
        end_mel = end_snap * self.snap_mel + self.half_audio_mel
        audio_data = audio_data / np.max(audio_data)
        audio_data = audio_data[:, start_mel:end_mel]

        # prepare label
        start_label = start_snap - first_ho_snap
        end_label = end_snap - start_snap + start_label
        beatmap_label = beatmap_label[start_label:end_label]

        if audio_data.shape[-1] != len(beatmap_label) * self.snap_mel + 2 * self.half_audio_mel:
            print('failed trim!')
            return None, None
        sample_data_list = []
        skip_sample = False
        for sample_idx in range(len(beatmap_label)):
            last_label = beatmap_label[sample_idx - 1] if sample_idx > 0 else 0
            start_mel = sample_idx * self.audio_mel
            feature = self.make_feature(
                audio_data[:, start_mel:start_mel + self.audio_mel],
                last_label,
                speed_stars,
                beatmap.bpm_min(),
            )
            sample_data_list.append(feature)
        if skip_sample:
            print('skipped sample!')
            return None, None
        sample_data_list = np.stack(sample_data_list)
        return sample_data_list, np.asarray(beatmap_label)

    def prepare(self, save=True):
        if self.random_seed is not None:
            random.seed(self.random_seed)
        table_name = 'FILTERED'
        ids = self.db.all_ids(table_name)
        # cond_data, label
        self.items = [[], []]
        data, label = self.items
        if self.take_first is not None:
            ids = ids[:self.take_first]
        print('ids')
        print(ids)
        cache = {0: None}
        for id_ in ids:
            record = self.db.get_record(id_, table_name)
            first_ho_snap = record[db.OsuDB.EXTRA_START_POS + 1]
            beatmap = pickle.loads(record[db.OsuDB.BEATMAP_POS])
            audio_path = os.path.join(self.audio_dir, record[db.OsuDB.AUDIOFILENAME_POS])
            # print('audio_path')
            # print(audio_path)
            if audio_path in cache:
                self.logger.debug('using cached %s' % audio_path)
                # print('using cached %s' % audio_path)
                audio_data = cache[audio_path]
            else:
                self.logger.debug('loaded %s' % audio_path)
                # print('loaded %s' % audio_path)
                with open(audio_path, 'rb') as f:
                    audio_data = pickle.load(f)
                cache[audio_path] = audio_data
                del cache[list(cache.keys())[0]]
            audio_data = audio_data.numpy()
            sample_data, sample_label = self.prepare_record(
                audio_data, beatmap, first_ho_snap
            )
            if sample_data is None:
                continue
            data.append(sample_data)
            label.append(sample_label)

        # max_value = np.max([np.max(seq_data[0]) for seq_data in cond_data])
        # for idx in range(len(cond_data)):
        #     cond_data[idx][0][:-label_num] = cond_data[idx][0][:-label_num] / max_value
        print([np.max(seq_data) for seq_data in data])
        print([np.min(seq_data) for seq_data in data])
        print('cond_data.shape')
        print([seq_data.shape for seq_data in data])
        print('label.shape')
        print([seq_label.shape for seq_label in label])
        # self.items[0] = np.stack(cond_data)
        # self.items[0] = np.stack(label)
        if save:
            self.save_raw()

    # def prepare_without_former_labels(self, save=True):
    #     if self.random_seed is not None:
    #         random.seed(self.random_seed)
    #     table_name = 'FILTERED'
    #     ids = self.db.all_ids(table_name)
    #     # cond_data, label
    #     self.items = [[], []]
    #     cond_data, label = self.items
    #     if self.take_first is not None:
    #         ids = ids[:self.take_first]
    #     print('ids')
    #     print(ids)
    #     cache = {0: None}
    #     for id_ in ids:
    #         record = self.db.get_record(id_, table_name)
    #         audio_path = os.path.join(self.audio_dir, record[db.OsuDB.AUDIOFILENAME_POS])
    #         # print('audio_path')
    #         # print(audio_path)
    #         if audio_path in cache:
    #             self.logger.debug('using cached %s' % audio_path)
    #             # print('using cached %s' % audio_path)
    #             audio_data = cache[audio_path]
    #         else:
    #             self.logger.debug('loaded %s' % audio_path)
    #             # print('loaded %s' % audio_path)
    #             with open(audio_path, 'rb') as f:
    #                 audio_data = pickle.load(f)
    #             cache[audio_path] = audio_data
    #             del cache[list(cache.keys())[0]]
    #         audio_data = audio_data.numpy()
    #         # to 1 channel
    #         audio_data = np.mean(audio_data, axis=0)
    #         crop_start_time = record[db.OsuDB.EXTRA_START_POS + 1]
    #         # print('crop_start_time %d' % crop_start_time)
    #         beatmap = pickle.loads(record[db.OsuDB.BEATMAP_POS])
    #         assert isinstance(beatmap, slider.Beatmap)
    #         snap_per_microsecond = beatmap_util.get_snap_per_microseconds(beatmap)
    #         total_snaps = beatmap_util.get_total_snaps(beatmap, self.snap_divisor)
    #         first_ho_time = beatmap_util.get_first_hit_object_time_microseconds(beatmap)
    #         # one label per snap
    #         # three classes by default
    #         beatmap_label = dataset_util.hitobjects_to_label(
    #             beatmap,
    #             first_ho_time,
    #             snap_per_microsecond,
    #             total_snaps,
    #             None,
    #             multi_label=True
    #         )
    #         # print('type(audio_data)')
    #         # print(type(audio_data))
    #         # calculate from beginning of cropped audio cond_data
    #         first_ho_snap = round((first_ho_time - crop_start_time) * snap_per_microsecond)
    #         # print('first_ho_snap')
    #         # print(first_ho_snap)
    #         start_snap = max(first_ho_snap, self.half_audio_snap, self.former_labels)
    #         # print('first_ho_snap')
    #         # print(first_ho_snap)
    #         start_label = start_snap - first_ho_snap
    #         start_mel = start_snap * self.snap_mel - self.half_audio_mel
    #
    #         end_snap = min(
    #             # snaps_from_start_label
    #             len(beatmap_label) + first_ho_snap,
    #             # available snaps in audio cond_data - self.half_audio_mel
    #             (audio_data.shape[1] - start_snap * self.snap_mel - self.half_audio_snap) // self.snap_mel + start_snap
    #         )
    #         end_label = end_snap - start_snap + start_label
    #         beatmap_label = beatmap_label[start_label:end_label]
    #         end_mel = end_snap * self.snap_mel + self.half_audio_mel
    #         before_crop_shape = audio_data.shape
    #         if end_mel > audio_data.shape[1]:
    #             print('before_crop_shape')
    #             print(before_crop_shape)
    #             print('end_mel')
    #             print(end_mel)
    #             continue
    #         audio_data = audio_data[:, start_mel:end_mel]
    #         sample_data_list = []
    #         for sample_idx in range(len(beatmap_label)):
    #             sample_start_mel = sample_idx*self.audio_mel
    #             sample_data = audio_data[:, sample_start_mel:sample_start_mel+self.audio_mel].reshape([-1])
    #             sample_data_list.append(sample_data)
    #         try:
    #             speed_stars = beatmap.speed_stars()
    #         except Exception:
    #             continue
    #         cond_data.append((np.stack(sample_data_list), speed_stars / self.coeff_speed_stars, beatmap.bpm_min() / self.coeff_bpm))
    #         label.append(np.asarray(beatmap_label))
    #
    #     print('cond_data.shape')
    #     print([seq_data[0].shape for seq_data in cond_data])
    #     print('label.shape')
    #     print([seq_label.shape for seq_label in label])
    #     if save:
    #         self.save_raw()
