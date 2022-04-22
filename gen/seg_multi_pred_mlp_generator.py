import os
import time
from datetime import timedelta
from itertools import chain

import slider
import torch
import yaml

from gen import gen_util
from gen.beatmap_generator import BeatmapGenerator
from gen.position_generator import RandomWalkInRectangle
from util import general_util, beatmap_util


class SegMultiLabelGenerator(BeatmapGenerator):
    def __init__(self,
                 inference_config_path='./resources/config/inference/seg_mlp_bi_lr0.1.yaml',
                 prepare_data_config_path='./resources/config/prepare_data/inference/seg_multi_label_data.yaml'):
        print('using inference_config_path %s' % inference_config_path)
        print('using prepare_data_config_path %s' % prepare_data_config_path)
        super().__init__(inference_config_path, prepare_data_config_path)
        data_arg = self.config_dict['data_arg']
        dataset_arg = data_arg['test_dataset_arg']
        self.beat_feature_frames = dataset_arg['beat_feature_frames'] if 'beat_feature_frames' in dataset_arg else 16384
        self.snap_divisor = dataset_arg['snap_divisor'] if 'snap_divisor' in dataset_arg else 8
        self.sample_beats = dataset_arg['sample_beats'] if 'sample_beats' in dataset_arg else 16
        self.pad_beats = dataset_arg['pad_beats'] if 'pad_beats' in dataset_arg else 4
        self.pad_frames = self.pad_beats * self.beat_feature_frames
        self.batch_size = data_arg['batch_size'] if 'batch_size' in dataset_arg else 1

    # def preprocess(self, audio_file_path):
    #     """
    #     Resampling and padding/trimming.
    #     """
    #     audio_data, sample_rate = audio_util.audioread_get_audio_data(audio_file_path)
    #     bpm, start_time, end_time = gen_util.extract_bpm(audio_file_path)
    #     start_frame, end_frame = SegMultiLabelDBDataset.get_aligned_start_end_frame(
    #         bpm, start_time, end_time, self.beat_feature_frames, self.sample_beats
    #     )
    #     resample_rate = round(self.beat_feature_frames * bpm / 60)  # * feature_frames_per_second
    #     resampled_audio_data = torchaudio.functional.resample(
    #         audio_data, sample_rate, resample_rate
    #     )
    #     data = SegMultiLabelDBDataset.preprocess(
    #         resampled_audio_data, start_frame, end_frame, self.pad_beats
    #     )
    #     return data
    def gen_hitobjects(self, beatmap, audio_label, start_time, snap_microseconds):
        audio_label = audio_label.numpy().tolist()

        def periods():
            """
            yields (current position, nonzero period, trailing zero period length)
            """
            pos = 0
            next_pos = 0
            stop = False
            while True:
                # skip prepending zeros
                while audio_label[next_pos] == 0:
                    next_pos += 1
                    if next_pos >= len(audio_label):
                        stop = True
                        break
                if stop:
                    break
                zero_num = next_pos - pos
                pos = next_pos
                while audio_label[next_pos] > 0:
                    next_pos += 1
                    if next_pos >= len(audio_label):
                        stop = True
                        break
                if stop:
                    break
                yield pos, audio_label[pos:next_pos], zero_num
                pos = next_pos

        # save some space about the border as circles have radius
        pos_gen = RandomWalkInRectangle(30, 482, 30, 354)
        pos_gen.move_to_random_pos()
        pos_gen.set_walk_dist_range(50, 200)
        # min_dist, max_dist = 0, -1
        # pin_dist = False

        ms_per_beat = 60000 / beatmap.bpm_min()
        for pos, period, trailing_zero_period_len in periods():
            time = (start_time + pos * snap_microseconds) / 1000
            if len(period) > 1:
                num_beats = len(period) / self.snap_divisor
                # we only generate linear sliders for simplicity
                pos_list = [
                    pos_gen.next_pos()
                    for _ in range(max(2, int(num_beats)))
                ]

                beatmap_util.add_slider(
                    beatmap, 'L', pos_list, time, num_beats, ms_per_beat
                )
                print(beatmap._hit_objects[-1].curve.points)
                print(('add slider at (%.3f, %s)' % (time, ' ({}, {})'*len(pos_list))).format(*chain(*pos_list)))
            else:
                x, y = pos_gen.next_pos()
                beatmap_util.add_circle(beatmap, (x, y), time)
                print(('add circle at (%.3f, (%d, %d))' % (time, x, y)))
            # pos_gen.set_walk_dist_range(5, 5)

    # def gen_hitobjects_at_time(self, beatmap, audio_label, start_time, snap_microseconds):
    #     # hitobject_time_list = [
    #     #     timedelta(microseconds=audio_info[1] + snap_idx * snap_microseconds)
    #     #     for snap_idx, snap_label in enumerate(audio_label) if snap_label == 1
    #     # ]
    #     # for hitobject_timedelta in hitobject_time_list:
    #     x_offset = [-25, 25]
    #     y_offset = [-25, 25]
    #     i, j = 0, 0
    #     for snap_idx, snap_label in enumerate(audio_label):
    #         if snap_label == 0:
    #             continue
    #         hitobject_timedelta = timedelta(microseconds=start_time + snap_idx * snap_microseconds)
    #         print('add %s at time %03f' % ('Circle', hitobject_timedelta / timedelta(seconds=1)))
    #         beatmap._hit_objects.append(
    #             slider.beatmap.Circle(
    #                 slider.Position(256 + x_offset[i], 192 + y_offset[j]),
    #                 hitobject_timedelta,
    #                 0
    #             )
    #         )
    #         i += 1
    #         if i >= len(x_offset):
    #             i = 0
    #         j += 1
    #         if j >= len(x_offset):
    #             j = 0

    def generate_beatmap(self, audio_file_path_list, speed_stars_list, out_path_list=None, audio_info_path=None, **kwargs):
        assert len(audio_file_path_list) == len(speed_stars_list)

        # prepare inference data
        print('preparing inference dara...')
        # list of (bpm, start_time, end_time)
        if audio_info_path is None:
            print('extracting bpm, start_time, end_time...')
            audio_info_list = [
                gen_util.extract_bpm(audio_file_path)
                for audio_file_path in audio_file_path_list
            ]
            audio_info_path = os.path.join(
                SegMultiLabelGenerator.DEFAULT_AUDIO_INFO_DIR,
                time.asctime(time.localtime())
                .replace(' ', '_')
                .replace(':', '_') + '.yaml'
            )
            audio_info_dict = {
                os.path.abspath(audio_file_path): {
                    'bpm': audio_info[0],
                    'start_time': audio_info[1],
                    'end_time': audio_info[2],
                } for audio_file_path, audio_info
                in zip(audio_file_path_list, audio_info_list)
            }
            print(audio_info_dict)
            with open(audio_info_path, 'wt') as f:
                yaml.dump(audio_info_dict, f)
            print('saved audio info to %s' % audio_info_path)
        else:
            with open(audio_info_path, 'rt') as f:
                audio_info_dict = yaml.load(f, Loader=yaml.FullLoader)
            audio_info_list = [
                list(audio_info_dict[os.path.abspath(audio_file_path)].values())
                for audio_file_path in audio_file_path_list
            ]
        beatmap_list = [
            beatmap_util.get_empty_beatmap()
            for _ in range(len(audio_file_path_list))
        ]
        for beatmap, audio_info, speed_stars in zip(beatmap_list, audio_info_list, speed_stars_list):
            beatmap_util.set_bpm(beatmap, audio_info[0], self.snap_divisor)
            # add two dummy hitobjects to pin down start_time, end_time
            beatmap_util.set_start_time(beatmap, timedelta(microseconds=audio_info[1]))
            beatmap_util.set_end_time(beatmap, timedelta(microseconds=audio_info[2]))
            beatmap.overall_difficulty = speed_stars
        self.data_preparer.prepare_inference_data(audio_file_path_list, beatmap_list)

        # run inference and get label for each audio
        print('running inference...')
        # into batch of size 1 to conform to the model input format
        label = self.inference.run_inference()
        print(label)
        audio_label_div = self.inference.data_iter.dataset.accumulate_audio_sample_num
        audio_label_div = [0] + audio_label_div
        audio_label_list = [
            label[audio_label_div[i-1]:audio_label_div[i]].reshape([-1])
            for i in range(1, len(audio_label_div))
        ]
        print(audio_label_list)

        # generate .osu files
        print('generating .osu from predicted labels')
        if out_path_list is None:
            out_path_list = [
                general_util.change_ext(path, '.osu')
                for path in audio_file_path_list
            ]
        else:
            assert len(out_path_list) == len(audio_file_path_list)
        for audio_label, beatmap, audio_info, out_path in \
                zip(audio_label_list, beatmap_list, audio_info_list, out_path_list):
            # in microseconds
            snap_microseconds = 60000000 / (audio_info[0] * self.snap_divisor)
            # clear formerly added two dummy hitobjects
            beatmap._hit_objects.clear()
            self.gen_hitobjects(beatmap, audio_label, audio_info[1], snap_microseconds)
            beatmap.write_path(out_path)
