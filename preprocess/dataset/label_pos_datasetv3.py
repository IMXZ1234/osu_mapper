import math
import os
import pickle
import random

import numpy as np
import slider
from matplotlib import pyplot as plt

from nn.dataset import dataset_util
from preprocess import db, prepare_data, filter
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
        # following five meta features are appended after every snap's mel features
        ho_density(number of hit_objects over total snap number, empirically ~N(0.125, 0.058)),
        blank_proportion(proportion of snaps which is not occupied by a hit_object, that is, where keyboard is up, empirically ~N(0.5, 0.07)),
        circle_proportion(num_circle/(num_circle+num_slider), empirically ~N(0.5, 0.15)),
        approach_rate('AR' in osu!),
        bpm,
    ]
    """
    def __init__(self, save_dir,
                 db_path=prepare_data.DEFAULT_TRAIN_MEL_DB_PATH,
                 audio_dir=prepare_data.DEFAULT_TRAIN_MEL_AUDIO_DIR,
                 audio_mel=4, snap_mel=4, snap_divisor=8,
                 take_first=100, random_seed=None,
                 coeff_approach_rate=10.,
                 coeff_bpm=120,
                 label_num=3,
                 switch_label=False,
                 density_label=False,
                 multibeat_label_fmt=0,
                 item_names=fit_dataset.FitDataset.DEFAULT_ITEM_NAMES, logger=None,
                 preprocess_arg=dict(),
                 separate_save=True,
                 **kwargs):
        """
        subseq_beat: length of subsequences.
        audio_mel: number of mel frames around target snap to be used as feature.
        label_num: zero is counted
        """
        super().__init__(save_dir, item_names, logger)
        self.db = db.OsuDB(db_path)
        self.audio_dir = audio_dir
        self.separate_save = separate_save

        self.take_first = take_first
        self.random_seed = random_seed

        self.audio_mel = audio_mel
        self.snap_mel = snap_mel
        self.snap_divisor = snap_divisor
        self.half_audio_mel = audio_mel // 2
        self.half_audio_snap = math.ceil(self.half_audio_mel / self.snap_mel)

        self.coeff_approach_rate = coeff_approach_rate
        self.coeff_bpm = coeff_bpm

        self.label_num = label_num
        self.switch_label = switch_label
        self.density_label = density_label
        self.multibeat_label_fmt = multibeat_label_fmt
        self.eye = np.eye(self.label_num)
        self.preprocess_arg = preprocess_arg

    def make_feature(self,
                     mel,
                     ho_density,
                     blank_proportion,
                     circle_proportion,
                     approach_rate,
                     bpm):
        """
        label of former snaps to one hot, and concatenate all features of this snap(sample) into a vector
        """
        mel = mel.reshape([-1])
        return np.concatenate([mel,
                               np.array([ho_density]),
                               np.array([blank_proportion]),
                               np.array([circle_proportion]),
                               np.array([approach_rate / self.coeff_approach_rate]),
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
        assert isinstance(beatmap, slider.Beatmap)
        snap_ms = beatmap_util.get_snap_milliseconds(beatmap)
        total_snaps = beatmap_util.get_total_snaps(beatmap, self.snap_divisor)
        first_ho_time = beatmap_util.get_first_hit_object_time_milliseconds(beatmap)
        # one label per snap
        # three classes by default
        try:
            beatmap_label = dataset_util.hitobjects_to_label_with_pos(
                beatmap,
                first_ho_time,
                snap_ms,
                total_snaps,
                None,
                multi_label=(self.label_num != 2),
                multibeat_label_fmt=self.multibeat_label_fmt,
            )
        except Exception:
            print('failed hitobjects_to_label_with_pos')
            return None, None
        if beatmap_label is None:
            return None, None
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
        ho_density, blank_proportion, circle_proportion = dataset_util.calculate_meta(beatmap, beatmap_label)
        skip_sample = False
        for start_mel in range(len(beatmap_label) * self.audio_mel):
            feature = self.make_feature(
                audio_data[:, start_mel:start_mel + 1],
                ho_density, blank_proportion, circle_proportion,
                beatmap.approach_rate,
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
        assert isinstance(beatmap, slider.Beatmap)
        start_snap = max(first_ho_snap, self.half_audio_snap)
        end_snap = (audio_data.shape[1] - start_snap * self.snap_mel - self.half_audio_mel) // self.snap_mel + start_snap
        start_mel = start_snap * self.snap_mel - self.half_audio_mel
        end_mel = end_snap * self.snap_mel + self.half_audio_mel
        audio_data = audio_data / np.max(audio_data)
        audio_data = audio_data[:, start_mel:end_mel]

        total_snaps = end_snap - start_snap + 1

        ho_density, blank_proportion, circle_proportion = dataset_util.sample_meta()
        sample_data_list = []
        for start_mel in range(total_snaps * self.audio_mel):
            feature = self.make_feature(
                audio_data[:, start_mel:start_mel + 1],
                ho_density, blank_proportion, circle_proportion,
                beatmap.approach_rate,  # approach_rate = np.random.uniform(0, 10)
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
            if self.separate_save:
                self.save_raw_separate()
                data_len_list = [len(d) // self.snap_mel for d in self.items[0]]
                with open(os.path.join(self.save_dir, 'info.pkl'), 'wb') as f:
                    pickle.dump(data_len_list, f)
            else:
                self.save_raw()

    def prepare(self, save=True):
        if self.random_seed is not None:
            random.seed(self.random_seed)
        # table_name = 'FILTERED'
        table_name = 'MAIN'
        ids = self.db.all_ids(table_name)
        # cond_data, label
        self.items = [[], []]
        data, label = self.items
        if self.take_first is not None:
            ids = ids[:self.take_first]
        print('ids')
        print(ids)
        cache = {0: None}
        sample_filter = filter.OsuTrainDataFilterGroup(
            [
                filter.ModeFilter(),
                filter.SnapDivisorFilter(),
                filter.SnapInNicheFilter(),
                filter.SingleUninheritedTimingPointFilter(),
                filter.SingleBMPFilter(),
                filter.BeatmapsetSingleAudioFilter(),
                filter.HitObjectFilter(),
            ]
        )
        sample_idx = 0
        data_len_list = []
        all_speed_stars = []
        all_difficulty = []
        for id_ in ids:
            record = self.db.get_record(id_, table_name)
            first_ho_snap = record[db.OsuDB.EXTRA_START_POS + 1]
            beatmap = pickle.loads(record[db.OsuDB.BEATMAP_POS])
            audio_path = os.path.join(self.audio_dir, record[db.OsuDB.AUDIOFILENAME_POS])
            if not sample_filter.filter(beatmap, audio_path):
                print('skipping %d, title %s' % (id_, beatmap.title))
                continue
            try:
                all_speed_stars.append(beatmap.speed_stars())
                all_difficulty.append(beatmap.overall_difficulty)
            except Exception:
                print('failed calculating difficulty')
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
                print('failed %d' % id_)
                continue
            else:
                print('success %d' % id_)
            if self.separate_save and save:
                data_len_list.append(len(sample_data) // self.snap_mel)
                self.save_raw_separate_sample_items([sample_data, sample_label], sample_idx)
            else:
                data.append(sample_data)
                label.append(sample_label)
            sample_idx += 1

        if self.separate_save and save:
            with open(os.path.join(self.save_dir, 'info.pkl'), 'wb') as f:
                pickle.dump(data_len_list, f)
        else:
            print(all_difficulty)
            print(all_speed_stars)
            plt.hist(all_difficulty)
            plt.show()
            plt.hist(all_speed_stars)
            plt.show()
            # max_value = np.max([np.max(seq_data[0]) for seq_data in cond_data])
            # for idx in range(len(cond_data)):
            #     cond_data[idx][0][:-label_num] = cond_data[idx][0][:-label_num] / max_value
            print([np.max(seq_data) for seq_data in data])
            print([np.min(seq_data) for seq_data in data])
            print('cond_data.shape')
            print([seq_data.shape for seq_data in data])
            print('label.shape')
            print([seq_label.shape for seq_label in label])
            print(label[0])
            # self.items[0] = np.stack(cond_data)
            # self.items[0] = np.stack(label)
            if save:
                self.save_raw()

    def prepare_from_songs_dir(self, save=True):
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
                print('failed %d' % id_)
                continue
            else:
                print('success %d' % id_)
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
        print(label[0])
        # self.items[0] = np.stack(cond_data)
        # self.items[0] = np.stack(label)
        if save:
            self.save_raw()
