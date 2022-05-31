import argparse
import os
import yaml

from gen.seg_multi_pred_mlp_generator import SegMultiLabelGenerator
from gen.mel_mlp_generator import MelGenerator
from gen.mlpv2_generator import MLPv2Generator
from gen.rnnv1_generator import RNNv1Generator

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
    generator = RNNv1Generator(
        r'./resources/config/inference/rnnv1_lr0.01.yaml',
        r'./resources/config/prepare_data/inference/rnn_data.yaml',
    )

    # r"C:\CloudMusic\羊腿腿Y.tui - 遠くの子守の唄 - 《无职转生》第一季part.2 OP之一（翻自 大原ゆい子）.mp3"
    audio_file_path = \
        r"E:\Games\ELDEN RING\[Hi-Res][211122]TVアニメ『無職転生 ～異世界行ったら本気だす～2』OP3主题歌「遠くの子守の唄」／大原ゆい子[96kHz／24bit][FLAC]/01.遠くの子守の唄.flac"
        # r"C:\Users\asus\AppData\Local\osu!\Songs\703718 Hiromi Sato - Jiyuu no Hane\audio.mp3"

    audio_info_path = \
    r'./resources/gen/audio_info\01.遠くの子守の唄.yaml'
    # None

    meta_list = [
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': '大原ゆい子',
            'artist': 'oohara yuiko',  # indispensable
            'title_unicode': '遠くの子守の唄',
            'title': 'tooku no komamo no uta',  # indispensable
            'version': 'Insane',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
        }
    ]
    # meta_list = [
    #     {
    #         'audio_filename': os.path.basename(audio_file_path),  # indispensable
    #         'artist_unicode': '佐藤ひろ美',
    #         'artist': 'Hiromi Sato',  # indispensable
    #         'title_unicode': '自由の翅',
    #         'title': 'Jiyuu no Hane',  # indispensable
    #         'version': 'Insane',  # indispensable
    #         'creator': 'IMXZ123',
    #         'circle_size': 3,
    #         'approach_rate': 8,
    #         'slider_tick_rate': 2,
    #     }
    # ]
    for meta in meta_list:
        save_meta(meta)
    generator.generate_beatmapsets([audio_file_path], [[2]], [meta_list], [None],
                                   audio_info_path_list=[audio_info_path])
