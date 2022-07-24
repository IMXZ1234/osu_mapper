import os
from abc import abstractmethod
from datetime import timedelta

import yaml

from nn import inference
from gen import gen_util
from preprocess import prepare_data_util, prepare_data
from util import general_util, beatmap_util


class BeatmapGenerator:
    def __init__(self, inference_config_path,
                 prepare_data_config_path=None):
        """
        Initialize DataPreparer and Inference.
        """
        self.inference_config_path = inference_config_path
        self.prepare_data_config_path = prepare_data_config_path
        with open(prepare_data_config_path, 'rt', encoding='utf-8') as f:
            self.prepare_data_config = yaml.load(f, Loader=yaml.FullLoader)
        self.data_preparer = prepare_data.DataPreparer(**self.prepare_data_config)
        with open(inference_config_path, 'rt', encoding='utf-8') as f:
            self.config_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.inference = inference.Inference(**self.config_dict)

    @staticmethod
    def get_bpm_start_end_time(audio_file_path, audio_info_path=None, title=None):
        if audio_info_path is None:
            print('extracting bpm, start_time, end_time...')
            audio_info = gen_util.extract_bpm(audio_file_path)
            audio_info_path = os.path.join(
                gen_util.DEFAULT_AUDIO_INFO_DIR,
                title + '.yaml' if title is not None else
                general_util.change_ext(os.path.basename(audio_file_path), '.yaml')
            )
            audio_info_dict = {
                'bpm': audio_info[0],
                'start_time': audio_info[1],
                'end_time': audio_info[2],
            }
            print(audio_info_dict)
            with open(audio_info_path, 'wt') as f:
                yaml.dump(audio_info_dict, f)
            print('saved audio info to %s' % audio_info_path)
        else:
            with open(audio_info_path, 'rt') as f:
                audio_info_dict = yaml.load(f, Loader=yaml.FullLoader)
            print(audio_info_dict)
            audio_info = [
                audio_info_dict['bpm'],
                audio_info_dict['start_time'],
                audio_info_dict['end_time'],
            ]
        return audio_info

    @staticmethod
    def initialize_beatmaps(audio_info, speed_stars_list, meta_list, snap_divisor=8):
        beatmap_list = [
            beatmap_util.get_empty_beatmap()
            for _ in range(len(speed_stars_list))
        ]
        for beatmap, speed_stars, meta in zip(beatmap_list, speed_stars_list, meta_list):
            beatmap_util.set_bpm(beatmap, audio_info[0], snap_divisor)
            beatmap.timing_points[0].offset = timedelta(milliseconds=audio_info[1])
            # add two dummy hitobjects to pin down start_time, end_time
            beatmap_util.set_start_time(beatmap, timedelta(milliseconds=audio_info[1]))
            beatmap_util.set_end_time(beatmap, timedelta(milliseconds=audio_info[2]))
            # use overall_difficulty field to record speed_stars for simplicity
            beatmap.overall_difficulty = speed_stars
            beatmap_util.set_meta(beatmap, meta)
        return beatmap_list

    @abstractmethod
    def generate_beatmapset(self, audio_file_path, speed_stars_list, meta_list,
                            osu_out_path_list=None, audio_info_path=None, audio_idx=0, **kwargs):
        """
        Generate a bunch of beatmap for one audio/beatmapset
        """
        raise NotImplementedError

    def generate_beatmapsets(self, audio_file_path_list, speed_stars_list_list, meta_list_list=None,
                             osu_out_path_list_list=None, audio_info_path_list=None, **kwargs):
        """
        Generate a bunch of beatmapsets for a bunch of audios
        """
        assert len(audio_file_path_list) == len(speed_stars_list_list)
        if audio_info_path_list is None:
            audio_info_path_list = [None] * len(audio_file_path_list)

        for audio_idx, (audio_file_path, speed_stars_list, meta_list, osu_out_path_list, audio_info_path) \
                in enumerate(zip(audio_file_path_list, speed_stars_list_list, meta_list_list, osu_out_path_list_list,
                                 audio_info_path_list)):
            self.generate_beatmapset(audio_file_path, speed_stars_list, meta_list,
                                     osu_out_path_list, audio_info_path, audio_idx, **kwargs)

    def save_to_osu_songs_dir(self,
                              audio_file_path_list,
                              osu_file_path_list,
                              osu_songs_dir=prepare_data_util.OsuSongsDir.DEFAULT_OSU_SONGS_DIR):
        for audio_file_path, osu_file_path in zip(audio_file_path_list, osu_file_path_list):
            pass
