import os
from datetime import timedelta

import slider

import inference
from gen.beatmap_generator import BeatmapGenerator
from gen.gen_util import extract_bpm
from util import beatmap_util


class CNNGenerator(BeatmapGenerator):
    def __init__(self, audio_file_path, inference_config_path, model_path, device='cpu'):
        super().__init__(audio_file_path)
        self.inference = inference.Inference(inference_config_path, model_path)

    def generate_beatmapset(self, audio_file_path_list, speed_stars_list,
                            osu_out_path_list=None, audio_info_path=None, **kwargs):
        if audio_info_path is not None:
            # load audio info (bpm, first_beat, last_beat) from file if possible
            if os.path.exists(audio_info_path):
                with open(audio_info_path, 'r') as f:
                    line = f.readline()
                    s = line.split(',')
                    bpm, first_beat, last_beat = [float(part) for part in s]
            else:
                bpm, first_beat, last_beat = extract_bpm(audio_file_path_list)
                with open(audio_info_path, 'w') as f:
                    f.write(','.join([str(bpm), str(first_beat), str(last_beat)]))
        else:
            bpm, first_beat, last_beat = extract_bpm(audio_file_path_list)

        print('bpm %f, first_beat %f, last_beat %f' % (bpm, first_beat, last_beat))
        label = self.inference.run_inference(audio_file_path_list, speed_stars_list, bpm,
                                             start_time=first_beat, end_time=last_beat)
        beatmap = self.label_to_beatmap(label, speed_stars_list, bpm, start_time=first_beat, end_time=last_beat)
        beatmap.write_path(osu_out_path_list)

    @staticmethod
    def label_to_beatmap(label, speed_stars, bpm, start_time, end_time, snap_divisor=8):
        beatmap = beatmap_util.get_empty_beatmap()
        beatmap.bpm_min = bpm
        beatmap.bpm_max = bpm
        beatmap.speed_stars = speed_stars
        ms_per_beat = 60000 / bpm
        snap_per_microsecond = bpm * snap_divisor / 60000000
        total_snap_num = len(label)
        print('label')
        print(label)
        print('total_snap_num')
        print(total_snap_num)
        print('total snap num')
        print((end_time - start_time) * snap_per_microsecond)
        left, right, top, bottom = 0, 640, 0, 480
        beatmap.timing_points.append(slider.beatmap.TimingPoint(
            timedelta(microseconds=start_time),
            ms_per_beat,
            0, 0, 0, 100, None, False
        ))
        for snap_idx, snap_result in enumerate(label):
            if snap_result == 1:
                beatmap._hit_objects.append(slider.beatmap.Circle(
                    slider.Position(0, 0),
                    timedelta(microseconds=snap_idx / snap_per_microsecond + start_time),
                    0))
        return beatmap
