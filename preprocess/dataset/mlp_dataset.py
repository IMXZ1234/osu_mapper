import math
import os
import pickle
import random

import numpy as np
import slider

from nn.dataset import dataset_util
from preprocess import db, prepare_data
from util import beatmap_util, audio_util, general_util
from preprocess.dataset import fit_dataset
from preprocess.preprocessor import MelPreprocessor


class MLPDataset(fit_dataset.FitDataset):
    DEFAULT_SAVE_DIR = r'./resources/cond_data/fit/rnn/'
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
                 audio_mel=4, snap_mel=4, snap_divisor=8,
                 former_labels=16,
                 take_first=100, random_seed=None,
                 coeff_speed_stars=2.5,
                 coeff_bpm=120,
                 label_num=3,
                 item_names=fit_dataset.FitDataset.DEFAULT_ITEM_NAMES, logger=None,
                 preprocess_arg=dict()):
        """
        subseq_beat: length of subsequences.
        audio_mel: number of mel frames around target snap to be used as feature.
        """
        super().__init__(save_dir, item_names, logger)
        self.db = db.OsuDB(db_path)
        self.audio_dir = audio_dir

        self.take_first = take_first
        self.random_seed = random_seed

        self.former_labels = former_labels
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

    def make_feature(self, mel, label, bpm, speed_stars):
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
                               np.asarray([speed_stars]) / self.coeff_speed_stars,
                               np.asarray([bpm]) / self.coeff_bpm])

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
        audio_data = np.mean(audio_data, axis=0)
        start_snap = max(snap_offset, self.half_audio_snap)
        start_mel = start_snap * self.snap_mel - self.half_audio_mel

        end_snap = (audio_data.shape[1] - start_snap * self.snap_mel - self.half_audio_mel) // self.snap_mel + start_snap
        end_mel = end_snap * self.snap_mel + self.half_audio_mel
        audio_data = audio_data[:, start_mel:end_mel]
        audio_data = audio_data / np.max(audio_data)
        return audio_data

    class MLPSampleDataGenerator:
        def __init__(self, dataset, audio_path, beatmap):
            self.dataset = dataset
            self.audio_data = dataset.preprocess_audio(audio_path, beatmap)
            self.beatmap = beatmap
            self.sample_num = len(self.audio_data) // self.dataset.audio_mel
            self.sample_idx = 0
            label_history = [1] * (self.dataset.former_labels // 2) + [0] * (self.dataset.former_labels // 2)
            random.shuffle(label_history)
            self.label_history = np.asarray(label_history)

        def __len__(self):
            return self.sample_num

        def get_next_sample_feature(self, last_label):
            if self.sample_idx >= self.sample_num:
                return None
            start_mel = self.sample_idx * self.dataset.audio_mel
            self.label_history = np.concatenate([self.label_history[1:], np.array([last_label])])
            feature = self.dataset.make_feature(
                self.audio_data[
                    :, start_mel:start_mel+self.dataset.audio_mel
                ],
                self.label_history,
                self.beatmap.bpm_min(),
                self.beatmap.speed_stars(),
            )
            print('feature.shape')
            print(feature.shape)
            print(feature)
            return feature

    def get_sample_data_generator(self, audio_path, beatmap):
        return self.MLPSampleDataGenerator(self, audio_path, beatmap)

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
        label_num = 3
        eye = np.eye(label_num)
        cache = {0: None}
        for id_ in ids:
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
            audio_data = audio_data.numpy()
            # to 1 channel
            audio_data = np.mean(audio_data, axis=0)
            crop_start_time = record[db.OsuDB.EXTRA_START_POS + 2]
            first_ho_snap = record[db.OsuDB.EXTRA_START_POS + 1]
            # print('crop_start_time %d' % crop_start_time)
            beatmap = pickle.loads(record[db.OsuDB.BEATMAP_POS])
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
            beatmap_label = np.asarray(beatmap_label)
            start_snap = max(first_ho_snap, self.half_audio_snap, self.former_labels)
            start_label = start_snap - first_ho_snap
            start_mel = start_snap * self.snap_mel - self.half_audio_mel

            end_snap = min(
                # snaps_from_start_label
                len(beatmap_label) + first_ho_snap,
                # available snaps in audio cond_data - self.half_audio_mel
                (audio_data.shape[1] - start_snap * self.snap_mel - self.half_audio_mel) // self.snap_mel + start_snap
            )
            end_label = end_snap - start_snap + start_label
            ori_beatmap_label = beatmap_label.copy()
            beatmap_label = beatmap_label[start_label:end_label]
            end_mel = end_snap * self.snap_mel + self.half_audio_mel
            audio_data = audio_data[:, start_mel:end_mel]
            audio_data = audio_data / np.max(audio_data)
            if audio_data.shape[-1] != len(beatmap_label) * self.snap_mel + 2 * self.half_audio_mel:
                continue
            sample_data_list = []
            skip_sample = False
            for sample_idx in range(len(beatmap_label)):
                sample_start_mel = sample_idx*self.audio_mel
                sample_data = audio_data[:, sample_start_mel:sample_start_mel+self.audio_mel]
                if len(sample_data) < self.audio_mel:
                    skip_sample = True
                    break
                sample_data_list.append(sample_data.reshape([-1]))
            if skip_sample:
                print('skipped sample!')
                continue
            try:
                speed_stars = beatmap.speed_stars()
            except Exception:
                continue
            sample_label_feature_list = []
            # print('len(beatmap_label)')
            # print(len(beatmap_label))
            # print('len(ori_beatmap_label)')
            # print(len(ori_beatmap_label))
            for sample_idx in range(len(beatmap_label)):
                sample_start_label = start_label-self.former_labels+sample_idx
                sample_former_labels = ori_beatmap_label[sample_start_label:sample_start_label+self.former_labels]
                if len(sample_former_labels) < self.former_labels:
                    padded_sample_former_labels = np.zeros([self.former_labels], dtype=int)
                    if len(sample_former_labels) > 0:
                        padded_sample_former_labels[-len(sample_former_labels):] = sample_former_labels
                    sample_former_labels = padded_sample_former_labels
                sample_label_feature = eye[sample_former_labels].reshape([-1])
                sample_label_feature_list.append(sample_label_feature)
            sample_data_list = np.stack(sample_data_list)
            sample_label_feature_list = np.stack(sample_label_feature_list)
            meta_feature_list = np.array(
                [speed_stars / self.coeff_speed_stars, beatmap.bpm_min() / self.coeff_bpm]
            )
            meta_feature_list = np.tile(meta_feature_list, [len(sample_data_list), 1])
            # print('meta_feature_list.shape')
            # print(meta_feature_list.shape)
            sample_data_list = np.concatenate([sample_data_list, sample_label_feature_list, meta_feature_list], axis=1)
            data.append(sample_data_list)
            label.append(beatmap_label)

        label = np.concatenate(label, axis=0)
        data = np.concatenate(data, axis=0)
        print(np.max(data))
        print(np.min(data))
        self.items[0] = data
        self.items[1] = label
        print('cond_data.shape')
        print(data.shape)
        print('label.shape')
        print(label.shape)
        if save:
            self.save_raw()
