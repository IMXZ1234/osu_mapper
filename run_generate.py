import argparse
import os
import yaml

from gen.seg_multi_pred_mlp_generator import SegMultiLabelGenerator
from gen.mel_mlp_generator import MelGenerator

from util import general_util


DEFAULT_META_DIR = r'./resources/gen/meta'


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', type=str, help=r'Path to audio file to convert.'
    )
    return parser


def save_meta(meta, save_dir=DEFAULT_META_DIR):
    meta_filename = general_util.change_ext(meta['audio_filename'], '.yaml')
    with open(os.path.join(save_dir, meta_filename), 'w') as f:
        yaml.dump(meta, f)


def load_meta(meta_filename, save_dir=DEFAULT_META_DIR):
    with open(os.path.join(save_dir, meta_filename), 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


if __name__ == '__main__':
    generator = MelGenerator(
        r'./resources/config/inference/mel_mlp_bi_lr0.1.yaml',
        r'./resources/config/prepare_data/inference/mel_data.yaml',
    )

    audio_file_path = r'C:\CloudMusic\森山良子 - 今日の日はさようなら.mp3'

    # audio_info_path = r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio_info\鷹石忍 - ジユウノハネ Piano arrangement.yaml'
    audio_info_path = None

    meta_list = [
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': '森山良子',
            'artist': 'moriyama ryoko',  # indispensable
            'title_unicode': '今日の日はさようなら',
            'title': 'kyo no hi ha sayounara',  # indispensable
            'version': 'Insane',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
        }
    ]
    for meta in meta_list:
        save_meta(meta)
    generator.generate_beatmapsets([audio_file_path], [[2]], [meta_list], [None],
                                   audio_info_path_list=[audio_info_path])
