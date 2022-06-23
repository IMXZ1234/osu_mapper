import os
import pickle
import random

import numpy as np
import slider

from nn.dataset import dataset_util
from preprocess import db, prepare_data
from util import beatmap_util, general_util
from preprocess.dataset import fit_dataset


class RecurDataset(fit_dataset.FitDataset):
    DEFAULT_SAVE_DIR = r'./resources/cond_data/fit/recur/'
    """
    A snap may be of label 0-5:
    We predict label of every snap with features:
    [
        one-hot label of classes of label_snap former snaps,
        mel features of audio_mel around target snap,
        speed_star,
        bpm,
    ]
    """
    def __init__(self, save_dir,
                 db_path=prepare_data.DEFAULT_TRAIN_MEL_DB_PATH,
                 audio_dir=prepare_data.DEFAULT_TRAIN_MEL_AUDIO_DIR,
                 label_snap=64, audio_mel=64, snap_mel=4, snap_divisor=8,
                 take_first=100, max_beatmap_sample_num=None, shuffle=False, random_seed=None,
                 keep_label_proportion=True,
                 item_names=fit_dataset.FitDataset.DEFAULT_ITEM_NAMES, logger=None):
        """
        label_snap: number of snaps before target snap whose label will be used as feature.
        audio_mel: number of mel frames around target snap to be used as feature.
        """
        super().__init__(save_dir, item_names, logger)
        self.db = db.OsuDB(db_path)
        self.audio_dir = audio_dir

        self.take_first = take_first
        self.max_beatmap_sample_num = max_beatmap_sample_num
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.keep_label_proportion = keep_label_proportion

        self.label_snap = label_snap
        self.audio_mel = audio_mel
        self.half_audio_mel = audio_mel // 2
        self.snap_mel = snap_mel
        self.snap_divisor = snap_divisor

    def make_feature(self, label, mel, bpm, speed_stars, class_num):
        """
        label of former snaps to one hot, and concatenate all features of this snap(sample) into a vector
        """
        label = np.asarray(label)
        one_hot_feature = np.eye(class_num)[label].reshape([-1])
        # print('mel.shape')
        # print(mel.shape)
        mel = mel.reshape([-1])
        return np.concatenate([one_hot_feature, mel, np.asarray([bpm]), np.asarray([speed_stars])])

    def prepare(self, save=True):
        if self.random_seed is not None:
            random.seed(self.random_seed)
        table_name = 'FILTERED'
        ids = self.db.all_ids(table_name)
        # cond_data, label
        self.items = [[], []]
        data, label = self.items
        label_num = 3
        if self.take_first is not None:
            ids = ids[:self.take_first]
        print('ids')
        print(ids)
        cache = {0: None}
        for id_ in ids:
            beatmap_sample_data = []
            beatmap_sample_label = []
            record = self.db.get_record(id_, table_name)
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
            crop_start_time = record[db.OsuDB.EXTRA_START_POS + 1]
            # print('crop_start_time %d' % crop_start_time)
            beatmap = pickle.loads(record[db.OsuDB.BEATMAP_POS])
            assert isinstance(beatmap, slider.Beatmap)
            snap_per_microsecond = beatmap_util.get_snap_per_microseconds(beatmap)
            total_snaps = beatmap_util.get_total_snaps(beatmap, self.snap_divisor)
            first_ho_time = beatmap_util.get_first_hit_object_time_microseconds(beatmap)
            # one label per snap
            # three classes by default
            beatmap_label = dataset_util.hitobjects_to_label(
                beatmap,
                first_ho_time,
                snap_per_microsecond,
                total_snaps,
                None,
                multi_label=True
            )
            first_ho_snap = (first_ho_time - crop_start_time) * snap_per_microsecond
            # print('first_ho_snap')
            # print(first_ho_snap)
            first_ho_snap = round(first_ho_snap)
            # print('first_ho_snap')
            # print(first_ho_snap)
            start_snap = max(first_ho_snap - self.label_snap, self.label_snap)
            start_label = start_snap + self.label_snap - first_ho_snap
            start_mel = start_snap * self.snap_mel - self.half_audio_mel
            beatmap_label = beatmap_label[start_label:]
            end_mel = len(beatmap_label) * self.snap_mel + start_mel
            before_crop_shape = audio_data.shape
            audio_data = audio_data[:, :, start_mel:end_mel]
            if audio_data.shape[-1] != len(beatmap_label) * self.snap_mel:
                print('before_crop_shape')
                print(before_crop_shape)
                print('after crop audio_data.shape')
                print(audio_data.shape)
                print('total_snaps')
                print(len(beatmap_label))
                print('total_snaps*snap_mel')
                print(len(beatmap_label)*self.snap_mel)
                print()
                continue
            beatmap_label = beatmap_label[start_label:]
            skip = False
            for i in range(len(beatmap_label)-start_label-self.label_snap-1):
                start_mel_pos = i * self.snap_mel
                try:
                    sample_x = self.make_feature(
                        beatmap_label[i:i+self.label_snap],
                        audio_data[:, :, start_mel_pos:start_mel_pos+self.audio_mel],
                        beatmap.bpm_min(),
                        beatmap.speed_stars(),
                        class_num=3
                    )
                except Exception:
                    skip = True
                    print('fail to calculate speed stars!')
                    break
                sample_y = beatmap_label[i+self.label_snap+1]
                beatmap_sample_data.append(sample_x)
                beatmap_sample_label.append(sample_y)
            if skip:
                continue
            if self.shuffle:
                random.shuffle(beatmap_sample_data)
                random.shuffle(beatmap_sample_label)
            if self.max_beatmap_sample_num is not None:
                if self.keep_label_proportion:
                    label_max_beatmap_sample_num = self.max_beatmap_sample_num // label_num
                    categorized = general_util.list_of_list_categorize(
                        [beatmap_sample_data, beatmap_sample_label],
                        list_pos_as_key=1, key_in_item=True
                    )
                    beatmap_sample_data, beatmap_sample_label = [], []
                    for k, v in categorized.items():
                        beatmap_sample_data.extend(v[0][:label_max_beatmap_sample_num])
                        beatmap_sample_label.extend(v[1][:label_max_beatmap_sample_num])
                else:
                    beatmap_sample_data = beatmap_sample_data[:self.max_beatmap_sample_num]
                    beatmap_sample_label = beatmap_sample_label[:self.max_beatmap_sample_num]
            data.extend(beatmap_sample_data)
            label.extend(beatmap_sample_label)
        if self.shuffle:
            random.shuffle(data)
            random.shuffle(label)
        self.items[0] = np.stack(data)
        print('cond_data.shape')
        print(self.items[0].shape)
        self.items[1] = np.asarray(label, dtype=int)
        print('label.shape')
        print(self.items[1].shape)
        if save:
            self.save_raw()
