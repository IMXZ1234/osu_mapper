import os
import pickle

import yaml
import numpy as np

from preprocess.dataset import mlp_dataset, rnn_dataset, rnn_nolabel_dataset, mlp_nolabel_dataset
from preprocess import db
from preprocess.prepare_data import (
    DEFAULT_TRAIN_AUDIO_DIR,
    DEFAULT_TRAIN_DB_PATH,
    DEFAULT_INFERENCE_AUDIO_DIR,
    DEFAULT_INFERENCE_DB_PATH,
    DEFAULT_BEATMAP_LIST_SAVE_PATH,

    prepare_inference_data_with_config,
    prepare_train_data_with_config,
)
from util import beatmap_util


def prepare_mel_train_data():
    config_path = r'./resources/config/prepare_data/train/mel_data.yaml'
    preprocessor_arg = {
        'preprocessor_type': 'preprocess.preprocessor.MelPreprocessor',
        'beat_feature_frames': 16384,
        'snap_offset': 0,
        'n_fft': 1024,
        'win_length': None,
        'hop_length': 512,
        'n_mels': 128,
        'snap_divisor': 8,
        'from_resampled': True
    }
    filter_arg = {
        'filter_type': 'preprocess.filter.OsuTrainDataFilterGroup',
        'filter_arg': [
            {
                'filter_type': 'preprocess.filter.HitObjectFilter',
            },
            {
                'filter_type': 'preprocess.filter.BeatmapsetSingleAudioFilter',
            },
            {
                'filter_type': 'preprocess.filter.SingleBMPFilter',
            },
            {
                'filter_type': 'preprocess.filter.SingleUninheritedTimingPointFilter',
            },
            {
                'filter_type': 'preprocess.filter.SnapDivisorFilter',
            },
            {
                'filter_type': 'preprocess.filter.SnapInNicheFilter',
            },
        ]
    }
    fold_divider_arg = {
        'fold_divider_type': 'preprocess.fold_divider.OsuTrainDBFoldDivider',
        # 'fold_divider_type': 'preprocess.fold_divider.OsuTrainDBDebugFoldDivider',
        'folds': 5,
        'shuffle': False,
    }
    prepare_data_config = {
        'preprocessor_arg': preprocessor_arg,
        'filter_arg': filter_arg,
        'fold_divider_arg': fold_divider_arg,
        'do_preprocess': True,
        'do_filter': True,
        'do_fold_divide': True,
        'save_dir': r'./resources/data/mel',
        'from_dir': DEFAULT_TRAIN_AUDIO_DIR,
        'db_path': r'./resources/data/osu_train_mel.db',
        'from_db_path': DEFAULT_TRAIN_DB_PATH,
        'save_ext': '.pkl',
    }
    with open(os.path.join(config_path), 'w') as f:
        yaml.dump(prepare_data_config, f)
    prepare_train_data_with_config(config_path)


def prepare_seg_multi_label_train_data():
    config_path = r'./resources/config/prepare_data/train/seg_multi_label_data.yaml'
    preprocessor_arg = {
        'preprocessor_type': 'preprocess.preprocessor.ResamplePreprocessor',
        'beat_feature_frames': 16384,
    }
    filter_arg = {
        'filter_type': 'preprocess.filter.OsuTrainDataFilterGroup',
        'filter_arg': [
            {
                'filter_type': 'preprocess.filter.HitObjectFilter',
            },
            {
                'filter_type': 'preprocess.filter.BeatmapsetSingleAudioFilter',
            },
            {
                'filter_type': 'preprocess.filter.SingleBMPFilter',
            },
            {
                'filter_type': 'preprocess.filter.SingleUninheritedTimingPointFilter',
            },
        ]
    }
    fold_divider_arg = {
        'fold_divider_type': 'preprocess.fold_divider.OsuTrainDBFoldDivider',
        # 'fold_divider_type': 'preprocess.fold_divider.OsuTrainDBDebugFoldDivider',
        'folds': 5,
        'shuffle': False,
    }
    prepare_data_config = {
        'preprocessor_arg': preprocessor_arg,
        'filter_arg': filter_arg,
        'fold_divider_arg': fold_divider_arg,
        'do_preprocess': False,
        'do_filter': False,
        'do_fold_divide': True,
        'save_dir': DEFAULT_TRAIN_AUDIO_DIR,
        'db_path': DEFAULT_TRAIN_DB_PATH,
    }
    with open(os.path.join(config_path), 'w') as f:
        yaml.dump(prepare_data_config, f)
    prepare_train_data_with_config(config_path)


def prepare_seg_multi_label_inference_data(from_audio_path_list, bpm_list, snap_divisor=8):
    assert len(from_audio_path_list) == len(bpm_list)
    beatmap_list = [beatmap_util.get_empty_beatmap()
                    for _ in range(len(from_audio_path_list))]
    for beatmap, bpm in zip(beatmap_list, bpm_list):
        beatmap_util.set_bpm(beatmap, bpm, snap_divisor)
    with open(DEFAULT_BEATMAP_LIST_SAVE_PATH, 'wb') as f:
        pickle.dump(beatmap_list, f)
    config_path = r'./resources/config/prepare_data/inference/seg_multi_label_data.yaml'
    preprocessor_arg = {
        'preprocessor_type': 'preprocess.preprocessor.ResamplePreprocessor',
        'beat_feature_frames': 16384,
    }
    prepare_data_config = {
        'preprocessor_arg': preprocessor_arg,
        'do_preprocess': False,
        'save_dir': DEFAULT_INFERENCE_AUDIO_DIR,
        'db_path': DEFAULT_INFERENCE_DB_PATH,
    }
    with open(os.path.join(config_path), 'w') as f:
        yaml.dump(prepare_data_config, f)
    prepare_inference_data_with_config(
        config_path, from_audio_path_list, DEFAULT_BEATMAP_LIST_SAVE_PATH
    )


def prepare_rnn_dataset():
    ds = rnn_dataset.RNNDataset(
        r'./resources/data/fit/rnn/snap_1',
        audio_mel=4,
        take_first=100,
        random_seed=404,
        coeff_speed_stars=2.5,
        coeff_bpm=120,
    )
    ds.prepare()
    ds.div_folds(save_first=1)


def prepare_rnn_nolabel_dataset():
    ds = rnn_nolabel_dataset.RNNNoLabelDataset(
        r'./resources/data/fit/rnn_nolabel',
        audio_mel=4,
        take_first=100,
        random_seed=404,
        coeff_speed_stars=2.5,
        coeff_bpm=120,
    )
    ds.prepare()
    ds.div_folds(save_first=1)


def prepare_mlp_dataset():
    ds = mlp_dataset.MLPDataset(
        r'./resources/data/fit/mlp/snap_1',
        audio_mel=4,
        take_first=100,
        random_seed=404,
        former_labels=16,
        coeff_speed_stars=2.5,
        coeff_bpm=120,
    )
    ds.prepare()
    ds.div_folds(save_first=1)


def prepare_mlp_nolabel_dataset():
    ds = mlp_nolabel_dataset.MLPNoLabelDataset(
        r'./resources/data/fit/mlp_nolabel',
        audio_mel=4,
        take_first=100,
        random_seed=404,
        coeff_speed_stars=2.5,
        coeff_bpm=120,
    )
    ds.prepare()
    ds.div_folds(save_first=1)


if __name__ == '__main__':
    db = db.OsuDB()
    # prepare_mel_train_data()
    # prepare_rnn_dataset()
    # data = rnn_dataset.RNNDataset(
    #     r'./resources/data/fit/rnn/snap_1',
    #     audio_mel=4,
    #     take_first=100,
    #     random_seed=404
    # ).get_raw_data()
    # print([np.max(seq_data[0]) for seq_data in data])
    # print([np.min(seq_data[0]) for seq_data in data])
    # database = db.OsuTrainDB(DEFAULT_DB_PATH)
    # # database.delete_view()
    # all_values = set(database.get_column('AUDIOFILENAME'))
    # print(all_values)
    # for value in set(database.get_column('AUDIOFILENAME')):
    #     if '_' in value:
    #         new_value = value.split('_')[0] + '.mp3'
    #         database.update_rows('AUDIOFILENAME', value, 'AUDIOFILENAME', new_value)
    # database.filter_data(filter.HitObjectFilter())
    # database.split_folds(fold_divider.OsuTrainDBFoldDivider(folds=5, shuffle=False))
    # EXTRA_COLUMNS = ['RESAMPLE_RATE']
    # new_columns = db.OsuTrainDB.DEFAULT_COLUMNS
    # new_columns.extend(EXTRA_COLUMNS)
    # database.delete_view(view_name=None)
    # database.alter_table_columns(new_columns=new_columns)
