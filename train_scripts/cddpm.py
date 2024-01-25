import yaml

from nn.train import train_with_config


def train_cddpm(setting_name='cddpm'):
    """
    acganv1_embedding
    """
    random_seed = 404
    # 32 beat per subseq
    beat_divisor = 8
    subseq_beats = 32
    subseq_snaps = subseq_beats * beat_divisor
    embedding_size = subseq_beats * 16
    label_size = 2 * subseq_snaps
    rnd_bank_size = max(embedding_size, label_size) * 1024
    mel_features = 40

    for lr in [0.0001]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': 'nn.net.diffusion.cuvit.CUViT',
            'dim': 48,
            'channels': 5,
            'num_meta': 2,
            'audio_in_channels': mel_features,
            'audio_out_channels': 12,
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': 'Adam',
            'lr': lr,
        }
        scheduler_arg = {
            'scheduler_type': 'MultiStepLR',
            'milestones': [5],
            'gamma': 1.,
        }
        data_arg = {'dataset': 'nn.dataset.feeder_onehot.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v6',
                         # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': None,
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': None,
                         'level_coeff': 0,
                         'item': 'coord_type',
                         'cache': 128,
                         },
                    'batch_size': 256,
                    'shuffle': False,
                    'num_workers': 8,
                    'drop_last': False}
        loss_arg = {
            'loss_type': 'MSELoss',
        }
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_step': 1,
                      'train_state_save_step': 1,
                      'log_batch_step': 512,
                      'log_epoch_step': 1,
                      }
        train_arg = {'epoch': 20480,
                     'eval_step': 1,
                     'start_epoch': 0,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,

                       # 'embedding_decoder': 'postprocess.embedding_decode.EmbeddingDecode',
                       # 'embedding_decoder_args': {
                       #     # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                       #     'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                       #     'beat_divisor': 8,
                       #     'subset_path': r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/counter.pkl',
                       # },
                       'output_device': 1,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'log_item': 'coord_type',
                       'train_type': 'cddpm',
                       'num_classes': 5,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 5},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',

                       'pred_objective': 'eps',
                       'noise_d': subseq_snaps,
                       'noise_d_low': None,
                       'noise_d_high': None,
                       'channels': 5,
                       'length': subseq_snaps,
                       'num_sample_steps': 500,
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)


def train_cddpm_datasetv7(setting_name='cddpm'):
    """
    acganv1_embedding
    """
    random_seed = 404
    # 32 beat per subseq
    beat_divisor = 8
    mel_frame_per_snap = 8
    subseq_beats = 64
    subseq_snaps = subseq_beats * beat_divisor
    embedding_size = subseq_beats * 16
    label_size = 2 * subseq_snaps
    rnd_bank_size = max(embedding_size, label_size) * 1024
    mel_features = 40

    for lr in [0.0001]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': 'nn.net.diffusion.cuvit.CUViT',
            'dim': 48,
            'channels': 5,
            'num_meta': 3,
            'audio_in_channels': mel_features,
            'audio_out_channels': 12,
            'audio_length_compress': mel_frame_per_snap,
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': 'Adam',
            'lr': lr,
        }
        scheduler_arg = {
            'scheduler_type': 'MultiStepLR',
            'milestones': [5],
            'gamma': 1.,
        }
        data_arg = {'dataset': 'nn.dataset.feeder_onehotv7.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v7',
                         # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': None,
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'mel_frame_per_snap': mel_frame_per_snap,
                         'rnd_bank_size': None,
                         'level_coeff': 0,
                         'item': 'coord_type',
                         'cache': 128,
                         },
                    'batch_size': 256,
                    'shuffle': False,
                    'num_workers': 8,
                    'drop_last': False}
        loss_arg = {
            'loss_type': 'MSELoss',
        }
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_step': 1,
                      'train_state_save_step': 1,
                      'log_batch_step': 512,
                      'log_epoch_step': 1,
                      }
        train_arg = {'epoch': 20480,
                     'eval_step': 1,
                     'start_epoch': 0,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,

                       # 'embedding_decoder': 'postprocess.embedding_decode.EmbeddingDecode',
                       # 'embedding_decoder_args': {
                       #     # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                       #     'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                       #     'beat_divisor': 8,
                       #     'subset_path': r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/counter.pkl',
                       # },
                       'output_device': 1,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'log_item': 'coord_type',
                       'train_type': 'cddpm',
                       'num_classes': 5,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 5},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',

                       'pred_objective': 'eps',
                       'noise_d': subseq_snaps,
                       'noise_d_low': None,
                       'noise_d_high': None,
                       'channels': 5,
                       'length': subseq_snaps,
                       'num_sample_steps': 500,
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)
