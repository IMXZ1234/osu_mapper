from abc import abstractmethod

import yaml

import inference
import prepare_data
from preprocess import prepare_data_util


class BeatmapGenerator:
    DEFAULT_AUDIO_INFO_DIR = './resources/gen/audio_info'

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

    @abstractmethod
    def generate_beatmap(self, audio_file_path_list, speed_stars_list,
                         out_path_list=None, audio_info_path=None, **kwargs):
        """
        audio_info_path may be specified
        """
        raise NotImplementedError

    def save_to_osu_songs_dir(self,
                              audio_file_path_list,
                              osu_file_path_list,
                              osu_songs_dir=prepare_data_util.OsuSongsDir.DEFAULT_OSU_SONGS_DIR):
        for audio_file_path, osu_file_path in zip(audio_file_path_list, osu_file_path_list):
            pass

