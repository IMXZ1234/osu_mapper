import argparse
import os
import yaml

from gen.seg_multi_pred_mlp_generator import SegMultiLabelGenerator
from gen.mel_mlp_generator import MelGenerator
from gen.mlpv2_generator import MLPv2Generator
# from gen.rnnv1_generator import RNNv1Generator
from gen.rnnv1_generator_test import RNNv1Generator
# from gen.rnnv3_generator import RNNGenerator
# from gen.rnnv4_generator import RNNGenerator
# from gen.cgan_generator import CGANGenerator
# from gen.seqganv1_generator import SeqGANv1Generator
from gen.seqganv3_generator import SeqGANv3Generator
# from gen.rnnv3_generator_test import RNNGenerator

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
    generator = SeqGANv3Generator()

    # r"C:\CloudMusic\羊腿腿Y.tui - 遠くの子守の唄 - 《无职转生》第一季part.2 OP之一（翻自 大原ゆい子）.mp3"
    audio_file_path = \
        r'C:\CloudMusic\FictionJunction - 暁の車.mp3'
        # r'D:\osu_mapper\resources\data\bgm\鷹石忍 - ジユウノハネ Piano arrangement.mp3'
        # r'C:\CloudMusic\森山良子 - 今日の日はさようなら.mp3'
        # r'C:\CloudMusic\忍 - 聖夜をめぐるまほう.mp3'
        # r'C:\Users\asus\AppData\Local\osu!\Songs\1013342 Sasaki Sayaka - Sakura, Reincarnation\audio.mp3'
        # r'C:\Users\asus\AppData\Local\osu!\Songs\1013342 Sasaki Sayaka - Sakura, Reincarnation\audio.mp3'
        # r'C:\Users\asus\AppData\Local\osu!\Songs\1766146 Nayugorou - White Promise\audio.ogg'
        # r"C:\CloudMusic\H ZETT M(Center),紅い流星(Left),まらしぃ - 君の知らない物语.mp3"
        # r"C:\Users\asus\AppData\Local\osu!\Songs\703718 Hiromi Sato - Jiyuu no Hane\audio.mp3"
        # r'C:\Users\asus\coding\python\osu_mapper\resources\cond_data\bgm\asdf.mp3'

    audio_info_path = \
        r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio_info\FictionJunction - 暁の車.yaml'
        # r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio_info\森山良子 - 今日の日はさようなら.yaml'
        # r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio_info\忍 - 聖夜をめぐるまほう.yaml'
        # r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio_info\Sakura, Reincarnation.yaml'
        # r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio_info\H ZETT M(Center),紅い流星(Left),まらしぃ - 君の知らない物语.yaml'
        # r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio_info\audio.yaml'
    # None
    # r'./resources/gen/audio_info\01.遠くの子守の唄.yaml'

    meta_list = [
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': 'FictionJunction',
            'artist': 'FictionJunction',  # indispensable
            'title_unicode': '暁の車',
            'title': 'akatsuki no kuruma',  # indispensable
            'version': 'Medium',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
        },
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': 'FictionJunction',
            'artist': 'FictionJunction',  # indispensable
            'title_unicode': '暁の車',
            'title': 'akatsuki no kuruma',  # indispensable
            'version': 'Hard',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
        },
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': 'FictionJunction',
            'artist': 'FictionJunction',  # indispensable
            'title_unicode': '暁の車',
            'title': 'akatsuki no kuruma',  # indispensable
            'version': 'Insane',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
        },
    ]
    # meta_list = [
    #     {
    #         'audio_filename': os.path.basename(audio_file_path),  # indispensable
    #         'artist_unicode': 'H ZETT M',
    #         'artist': 'H ZETT M',  # indispensable
    #         'title_unicode': '君の知らない物语',
    #         'title': 'kimi no shiranai monokatari',  # indispensable
    #         'version': 'Insane',  # indispensable
    #         'creator': 'IMXZ123',
    #         'circle_size': 3,
    #         'approach_rate': 8,
    #         'slider_tick_rate': 2,
    #     }
    # ]
    # meta_list = [
    #     {
    #         'audio_filename': os.path.basename(audio_file_path),  # indispensable
    #         'artist_unicode': '佐藤ひろ美',
    #         'artist': 'Hiromi Sato',  # indispensable
    #         'title_unicode': '自由の翅',
    #         'title': 'Jiyuu no Hane',  # indispensable
    #         'version': 'Normal',  # indispensable
    #         'creator': 'IMXZ123',
    #         'circle_size': 3,
    #         'approach_rate': 8,
    #         'slider_tick_rate': 2,
    #     },
    #     {
    #         'audio_filename': os.path.basename(audio_file_path),  # indispensable
    #         'artist_unicode': '佐藤ひろ美',
    #         'artist': 'Hiromi Sato',  # indispensable
    #         'title_unicode': '自由の翅',
    #         'title': 'Jiyuu no Hane',  # indispensable
    #         'version': 'Hard',  # indispensable
    #         'creator': 'IMXZ123',
    #         'circle_size': 3,
    #         'approach_rate': 8,
    #         'slider_tick_rate': 2,
    #     },
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
    #     },
    # ]
    # meta_list = [
    #     {
    #         'audio_filename': os.path.basename(audio_file_path),  # indispensable
    #         'artist_unicode': '森川良子',
    #         'artist': 'Hiromi Sato',  # indispensable
    #         'title_unicode': '今日の日はさようなら',
    #         'title': 'Jiyuu no Hane',  # indispensable
    #         'version': 'Medium',  # indispensable
    #         'creator': 'IMXZ123',
    #         'circle_size': 3,
    #         'approach_rate': 8,
    #         'slider_tick_rate': 2,
    #     },
    #     {
    #         'audio_filename': os.path.basename(audio_file_path),  # indispensable
    #         'artist_unicode': '森川良子',
    #         'artist': 'Hiromi Sato',  # indispensable
    #         'title_unicode': '今日の日はさようなら',
    #         'title': 'Jiyuu no Hane',  # indispensable
    #         'version': 'Hard',  # indispensable
    #         'creator': 'IMXZ123',
    #         'circle_size': 3,
    #         'approach_rate': 8,
    #         'slider_tick_rate': 2,
    #     },
    #     {
    #         'audio_filename': os.path.basename(audio_file_path),  # indispensable
    #         'artist_unicode': '森川良子',
    #         'artist': 'Hiromi Sato',  # indispensable
    #         'title_unicode': '今日の日はさようなら',
    #         'title': 'Jiyuu no Hane',  # indispensable
    #         'version': 'Insane',  # indispensable
    #         'creator': 'IMXZ123',
    #         'circle_size': 3,
    #         'approach_rate': 8,
    #         'slider_tick_rate': 2,
    #     },
    # ]
    # meta_list = [
    #     {
    #         'audio_filename': os.path.basename(audio_file_path),  # indispensable
    #         'artist_unicode': '忍',
    #         'artist': 'shinobu',  # indispensable
    #         'title_unicode': '聖夜をめぐるまほう',
    #         'title': 'seiya o meguru mahou',  # indispensable
    #         'version': 'Medium',  # indispensable
    #         'creator': 'IMXZ123',
    #         'circle_size': 3,
    #         'approach_rate': 8,
    #         'slider_tick_rate': 2,
    #     },
    #     {
    #         'audio_filename': os.path.basename(audio_file_path),  # indispensable
    #         'artist_unicode': '忍',
    #         'artist': 'shinobu',  # indispensable
    #         'title_unicode': '聖夜をめぐるまほう',
    #         'title': 'seiya o meguru mahou',  # indispensable
    #         'version': 'Hard',  # indispensable
    #         'creator': 'IMXZ123',
    #         'circle_size': 3,
    #         'approach_rate': 8,
    #         'slider_tick_rate': 2,
    #     },
    #     {
    #         'audio_filename': os.path.basename(audio_file_path),  # indispensable
    #         'artist_unicode': '忍',
    #         'artist': 'shinobu',  # indispensable
    #         'title_unicode': '聖夜をめぐるまほう',
    #         'title': 'seiya o meguru mahou',  # indispensable
    #         'version': 'Insane',  # indispensable
    #         'creator': 'IMXZ123',
    #         'circle_size': 3,
    #         'approach_rate': 8,
    #         'slider_tick_rate': 2,
    #     },
    # ]
    # meta_list = [
    #     {
    #         'audio_filename': os.path.basename(audio_file_path),  # indispensable
    #         'artist_unicode': 'Nayugorou',
    #         'artist': 'Nayugorou',  # indispensable
    #         'title_unicode': 'White Promise',
    #         'title': 'White Promise',  # indispensable
    #         'version': 'Insane',  # indispensable
    #         'creator': 'IMXZ123',
    #         'circle_size': 2,
    #         'approach_rate': 8,
    #         'slider_tick_rate': 2,
    #     }
    # ]
    # meta_list = [
    #     {
    #         'audio_filename': os.path.basename(audio_file_path),  # indispensable
    #         'artist_unicode': 'Hana',
    #         'artist': 'Hana',  # indispensable
    #         'title_unicode': 'Sakura, Reincarnation_2',
    #         'title': 'Sakura, Reincarnation',  # indispensable
    #         'version': 'Hard',  # indispensable
    #         'creator': 'IMXZ123',
    #         'circle_size': 3,
    #         'approach_rate': 8,
    #         'slider_tick_rate': 2,
    #     },
    #     {
    #         'audio_filename': os.path.basename(audio_file_path),  # indispensable
    #         'artist_unicode': 'Hana',
    #         'artist': 'Hana',  # indispensable
    #         'title_unicode': 'Sakura, Reincarnation_2',
    #         'title': 'Sakura, Reincarnation',  # indispensable
    #         'version': 'Insane',  # indispensable
    #         'creator': 'IMXZ123',
    #         'circle_size': 3,
    #         'approach_rate': 8,
    #         'slider_tick_rate': 2,
    #     }
    # ]
    for meta in meta_list:
        save_meta(meta)
    generator.generate_beatmapsets([audio_file_path], [[2.5, 3., 3.5]], [meta_list], [None],
                                   audio_info_path_list=[audio_info_path])
