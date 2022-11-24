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


class LabelPosDataset(fit_dataset.FitDataset):
    DEFAULT_SAVE_DIR = r'./resources/data/fit/rnn/'
    """
    A snap may be of label 0-5:
    We predict label of every snap with features:
    [
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
                 switch_label=False,
                 density_label=False,
                 multibeat_label_fmt=0,
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
        self.switch_label = switch_label
        self.density_label = density_label
        self.multibeat_label_fmt = multibeat_label_fmt
        self.eye = np.eye(self.label_num)
        self.preprocess_arg = preprocess_arg

    def make_feature(self, mel, speed_stars, bpm):
        """
        label of former snaps to one hot, and concatenate all features of this snap(sample) into a vector
        """
        mel = mel.reshape([-1])
        return np.concatenate([mel,
                               np.array([speed_stars / self.coeff_speed_stars]),
                               np.array([bpm / self.coeff_bpm]),
                               ])

    def preprocess_audio(self, audio_path, beatmap):
        # to 1 channel
        preprocessor = MelPreprocessor(**self.preprocess_arg)
        path_to = r'./resources/data/temp/%s' % general_util.change_ext(os.path.basename(audio_path), 'pkl')
        path_to_snap_offset = r'./resources/data/temp/snap_offset_%s' % general_util.change_ext(os.path.basename(audio_path), 'pkl')
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
        beatmap_label = dataset_util.hitobjects_to_label_with_pos(
            beatmap,
            first_ho_time,
            snap_ms,
            total_snaps,
            None,
            multi_label=(self.label_num != 2),
            multibeat_label_fmt=self.multibeat_label_fmt,
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
            start_mel = sample_idx * self.audio_mel
            feature = self.make_feature(
                audio_data[:, start_mel:start_mel + self.audio_mel],
                speed_stars,
                beatmap.bpm_min(),
            )
            sample_data_list.append(feature)
        if skip_sample:
            print('skipped sample!')
            return None, None
        sample_data_list = np.stack(sample_data_list)
        return sample_data_list, np.asarray(beatmap_label)

    def prepare_record_inference(self, audio_data, beatmap, first_ho_snap):
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
            print('use overall_difficulty as speed_stars in inference')
            speed_stars = beatmap.overall_difficulty
        assert isinstance(beatmap, slider.Beatmap)
        start_snap = max(first_ho_snap, self.half_audio_snap)
        end_snap = (audio_data.shape[1] - start_snap * self.snap_mel - self.half_audio_mel) // self.snap_mel + start_snap
        start_mel = start_snap * self.snap_mel - self.half_audio_mel
        end_mel = end_snap * self.snap_mel + self.half_audio_mel
        audio_data = audio_data / np.max(audio_data)
        audio_data = audio_data[:, start_mel:end_mel]

        total_snaps = end_snap - start_snap + 1

        sample_data_list = []
        for sample_idx in range(total_snaps):
            start_mel = sample_idx * self.audio_mel
            feature = self.make_feature(
                audio_data[:, start_mel:start_mel + self.audio_mel],
                speed_stars,
                beatmap.bpm_min(),
            )
            sample_data_list.append(feature)
        sample_data_list = np.stack(sample_data_list)
        return sample_data_list

    def prepare_inference(self, audio_path, beatmap_list, save=True):
        self.items = [[]]
        for beatmap in beatmap_list:
            audio_data, snap_offset = self.preprocess_audio(audio_path, beatmap)
            # audio_data, _ = self.trim_audio_data(audio_data, snap_offset)
            # audio_data = audio_data.numpy()
            sample_data = self.prepare_record_inference(
                audio_data, beatmap, snap_offset
            )
            self.items[0].append(sample_data)
        if save:
            self.save_raw()

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
