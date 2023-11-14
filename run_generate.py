import argparse
import os

import yaml

from util import general_util


DEFAULT_GEN_DIR = r'./resources/gen'
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


def test_gen_1(generator):
    audio_file_path = \
        r'C:\Users\asus\AppData\Local\osu!\Songs\1013342 Sasaki Sayaka - Sakura, Reincarnation\audio.mp3'

    audio_info_path = \
        r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio_info\Sakura, Reincarnation.yaml'

    meta_list = [
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': 'Hana',
            'artist': 'Hana',  # indispensable
            'title_unicode': 'Sakura, Reincarnation',
            'title': 'Sakura, Reincarnation',  # indispensable
            'version': 'Easy',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
        },
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': 'Hana',
            'artist': 'Hana',  # indispensable
            'title_unicode': 'Sakura, Reincarnation',
            'title': 'Sakura, Reincarnation',  # indispensable
            'version': 'Hard',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
        },
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': 'Hana',
            'artist': 'Hana',  # indispensable
            'title_unicode': 'Sakura, Reincarnation',
            'title': 'Sakura, Reincarnation',  # indispensable
            'version': 'Insane',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
        },
    ]
    for meta in meta_list:
        save_meta(meta)
    generator.generate_beatmapsets([audio_file_path], [[2.5, 3., 3.5]], [meta_list], [None],
                                   audio_info_path_list=[audio_info_path])


def test_gen_2(generator):
    audio_file_path = \
        r'C:\CloudMusic\忍 - 聖夜をめぐるまほう.mp3'

    audio_info_path = \
        r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio_info\忍 - 聖夜をめぐるまほう.yaml'

    meta_list = [
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': '忍',
            'artist': 'shinobu',  # indispensable
            'title_unicode': '聖夜をめぐるまほう',
            'title': 'seiya o meguru mahou',  # indispensable
            'version': 'Medium',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
        },
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': '忍',
            'artist': 'shinobu',  # indispensable
            'title_unicode': '聖夜をめぐるまほう',
            'title': 'seiya o meguru mahou',  # indispensable
            'version': 'Hard',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
        },
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': '忍',
            'artist': 'shinobu',  # indispensable
            'title_unicode': '聖夜をめぐるまほう',
            'title': 'seiya o meguru mahou',  # indispensable
            'version': 'Insane',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
        },
    ]
    for meta in meta_list:
        save_meta(meta)
    generator.generate_beatmapsets([audio_file_path], [[2.5, 3., 3.5]], [meta_list], [None],
                                   audio_info_path_list=[audio_info_path])


def test_gen_3(generator):
    audio_file_path = \
        r'C:\CloudMusic\森山良子 - 今日の日はさようなら.mp3'

    audio_info_path = \
        r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio_info\森山良子 - 今日の日はさようなら.yaml'

    meta_list = [
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': '森川良子',
            'artist': 'Hiromi Sato',  # indispensable
            'title_unicode': '今日の日はさようなら',
            'title': 'Jiyuu no Hane',  # indispensable
            'version': 'Medium',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
        },
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': '森川良子',
            'artist': 'Hiromi Sato',  # indispensable
            'title_unicode': '今日の日はさようなら',
            'title': 'Jiyuu no Hane',  # indispensable
            'version': 'Hard',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
        },
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': '森川良子',
            'artist': 'Hiromi Sato',  # indispensable
            'title_unicode': '今日の日はさようなら',
            'title': 'Jiyuu no Hane',  # indispensable
            'version': 'Insane',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
        },
    ]
    for meta in meta_list:
        save_meta(meta)
    generator.generate_beatmapsets([audio_file_path], [[2.5, 3., 3.5]], [meta_list], [None],
                                   audio_info_path_list=[audio_info_path])


def test_gen_4(generator):
    audio_file_path = \
        r'C:\CloudMusic\FictionJunction - 暁の車.mp3'
        # r'D:\osu_mapper\resources\data\bgm\鷹石忍 - ジユウノハネ Piano arrangement.mp3'
        # r'C:\Users\asus\AppData\Local\osu!\Songs\1766146 Nayugorou - White Promise\audio.ogg'
        # r"C:\CloudMusic\H ZETT M(Center),紅い流星(Left),まらしぃ - 君の知らない物语.mp3"
        # r"C:\Users\asus\AppData\Local\osu!\Songs\703718 Hiromi Sato - Jiyuu no Hane\audio.mp3"

    audio_info_path = \
        r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio_info\FictionJunction - 暁の車.yaml'
        # r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio_info\H ZETT M(Center),紅い流星(Left),まらしぃ - 君の知らない物语.yaml'
        # r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio_info\audio.yaml'

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
    for meta in meta_list:
        save_meta(meta)
    generator.generate_beatmapsets([audio_file_path], [[2.5, 3., 3.5]], [meta_list], [None],
                                   audio_info_path_list=[audio_info_path])


def test_gen_5(generator):
    audio_file_path = \
        r"C:\Users\asus\AppData\Local\osu!\Songs\703718 Hiromi Sato - Jiyuu no Hane\audio.mp3"

    audio_info_path = \
        r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio_info\Jiyuu no Hane.yaml'

    meta_list = [
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': '佐藤ひろ美',
            'artist': 'Hiromi Sato',  # indispensable
            'title_unicode': '自由の翅',
            'title': 'Jiyuu no Hane',  # indispensable
            'version': 'Normal',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
            'star': 2.5,
        },
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': '佐藤ひろ美',
            'artist': 'Hiromi Sato',  # indispensable
            'title_unicode': '自由の翅',
            'title': 'Jiyuu no Hane',  # indispensable
            'version': 'Hard',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
            'star': 4.5,
        },
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': '佐藤ひろ美',
            'artist': 'Hiromi Sato',  # indispensable
            'title_unicode': '自由の翅',
            'title': 'Jiyuu no Hane',  # indispensable
            'version': 'Insane',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
            'star': 5.5,
        },
    ]
    for meta in meta_list:
        save_meta(meta)
    generator.generate_beatmapset(
        os.path.join(DEFAULT_GEN_DIR, r'osz\test5.osz'),
        audio_file_path,
        meta_list,
        audio_info_path=audio_info_path,
        title='自由の翅',
        osu_dir=os.path.join(DEFAULT_GEN_DIR, r'osu\test5'),
    )


def test_gen_6(generator):
    audio_file_path = \
        os.path.join(DEFAULT_GEN_DIR, r"遠くの子守の唄.mp3")

    audio_info_path = \
        os.path.join(DEFAULT_GEN_DIR, r'audio_info\遠くの子守の唄.yaml')

    meta_list = [
        {
            'audio_filename': os.path.basename(audio_file_path),  # indispensable
            'artist_unicode': 'Yuiko',
            'artist': 'Yuiko',  # indispensable
            'title_unicode': '遠くの子守の唄',
            'title': 'tooku no komamo no uta',  # indispensable
            'version': '8',  # indispensable
            'creator': 'IMXZ123',
            'circle_size': 3,
            'approach_rate': 8,
            'slider_tick_rate': 2,
            'star': 4.5,
        },
    ]
    # for meta in meta_list:
    #     save_meta(meta)
    generator.generate_beatmapset(
        os.path.join(DEFAULT_GEN_DIR, r'osz\test6.osz'),
        audio_file_path,
        meta_list,
        audio_info_path=audio_info_path,
        title='遠くの子守の唄',
        osu_dir=os.path.join(DEFAULT_GEN_DIR, r'osu\test6'),
    )


if __name__ == '__main__':
    from gen.acgan_embedding_generator import ACGANEmbeddingGenerator
    # from gen.acgan_embeddingv1_generator import ACGANEmbeddingGenerator
    # import pickle
    # with open(r'C:\Users\asus\coding\python\osu_mapper\resources\data\processed_v4\mel\999834.pkl', 'rb') as f:
    #     print(pickle.load(f))
    generator = ACGANEmbeddingGenerator()
    # test_gen_6(generator)
    test_gen_5(generator)
#