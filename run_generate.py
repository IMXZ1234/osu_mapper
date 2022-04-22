import argparse

from gen.seg_multi_pred_mlp_generator import SegMultiLabelGenerator


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', type=str, help=r'Path to audio file to convert.'
    )
    return parser


if __name__ == '__main__':
    generator = SegMultiLabelGenerator(
        r'./resources/config/inference/seg_mlp_bi_lr0.1.yaml',
        r'./resources/config/prepare_data/inference/seg_multi_label_data.yaml',
    )
    # audio_file_path = r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\bgm\audio.mp3'
    # audio_file_path = r'C:\Users\asus\AppData\Local\osu!\Songs\869801 Meramipop - Sacrifice\audio.mp3'
    audio_file_path = r'C:\Users\asus\coding\python\osu_mapper\resources\gen\audio.mp3'
    out_path = r'.\resources\gen\out.osu'
    # file which saves bpm, start_time, end_time of audios in audio_file_path
    # audio_info_path = None
    audio_info_path = r'.\resources\gen\audio_info\Fri_Apr_22_20_23_40_2022.yaml'
    generator.generate_beatmap([audio_file_path], [0.3], [out_path], audio_info_path=audio_info_path)
