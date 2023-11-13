import yaml
import numpy as np

from nn.train import train_with_config


def train_acgan_embeddingv6_only_coord(setting_name='acgan_embeddingv6_only_coord'):
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

    for gen_lr, dis_lr in [[0.00003, 0.00003]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.acgan_embeddingv6_only_coord.Generator', 'nn.net.acgan_embeddingv6_only_coord.Discriminator'],
            'params': [
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'noise_dim': 128,
                    'norm': 'LN',
                    'middle_dim': 128,
                    'label_preprocess_dim': 1,
                    'audio_preprocess_dim': 8,
                    'condition_coeff': 1.,
                    # star
                    'cls_label_dim': 1,
                },
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'norm': 'LN',
                    'middle_dim': 128,
                    'preprocess_dim': 32,
                    # star
                    'cls_label_dim': 1,
                    'validity_sigmoid': False,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['RMSprop', 'RMSprop'],
            'params': [
                {'lr': gen_lr,},
                {'lr': dis_lr,},
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['MultiStepLR', 'MultiStepLR'],
            'params': [
                # {'milestones': [150, 300, 450], 'gamma': 1},
                # {'milestones': [150, 300, 450], 'gamma': 1},
                {'milestones': [5], 'gamma': 0.3},
                {'milestones': [5], 'gamma': 0.3},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.feeder_embedding.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v5',
                         'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': 10240,
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': None,
                         'level_coeff': 5,
                         'item': 'coord',
                         },
                    'batch_size': 256,
                    'shuffle': True,
                    'num_workers': 8,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': None},
                {'weight': None},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_step': 1,
                      'train_state_save_step': 1,
                      }
        train_arg = {'epoch': 20480,
                     'eval_step': 1,
                     'phases': ['pre_gen', 'pre_dis', 'adv', ],
                     'phase_epochs': [0, 0, 20480],
                     'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 0,
                     'generator_pretrain_epoch': 0,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [1000],
                     'gen_lambda': [1, 1],
                     # 'noise_level_step': [30, 60, 90, 120],
                     # 'noise_level_step': [30],
                     'noise_level_step': [15, 30, 45, 60, 75],
                     'noise_level': [1, 0.8, 0.6, 0.4, 0.2, 0.],
                     # 'noise_level': [10, 10, 5, 5, 5],
                     # 'noise_level': [5, 2.5, 1, 0.5, 0.25],
                     # 'noise_level': [0.5, 0.3, 0.1, 0.05, 0.],
                     'period': 15,
                     'lambda_gp': 10,
                     'lambda_cls': 1.,
                     'gp_type': 'lp',
                     # 'last_epoch': 12,
                     # 'last_phase': 'adversarial',
                     # 'last_epoch': 3,
                     # 'last_phase': 'pretrain_discriminator',
                     'start_epoch': 0,
                     # 'last_phase': 'pretrain_generator',
                     'start_phase': 'adv',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,

                       'embedding_decoder': 'postprocess.embedding_decode.EmbeddingDecode',
                       'embedding_decoder_args': {
                           # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                           'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                           'beat_divisor': 8,
                           'subset_path': r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/counter.pkl',
                       },
                       'output_device': 0,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'log_item': 'coord',
                       'train_type': 'acwganwb',
                       'num_classes': 5,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 5},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)


def train_acgan_embeddingv6_only_embedding(setting_name='acgan_embeddingv6_only_embedding'):
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

    for gen_lr, dis_lr in [[0.00003, 0.00003]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.acgan_embeddingv6_only_embedding.Generator', 'nn.net.acgan_embeddingv6_only_embedding.Discriminator'],
            'params': [
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'noise_dim': 128,
                    'norm': 'LN',
                    'middle_dim': 128,
                    'label_preprocess_dim': 1,
                    'audio_preprocess_dim': 8,
                    'condition_coeff': 1.,
                    # star
                    'cls_label_dim': 1,
                },
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'norm': 'LN',
                    'middle_dim': 128,
                    'preprocess_dim': 32,
                    # star
                    'cls_label_dim': 1,
                    'validity_sigmoid': False,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['RMSprop', 'RMSprop'],
            'params': [
                {'lr': gen_lr,},
                {'lr': dis_lr,},
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['MultiStepLR', 'MultiStepLR'],
            'params': [
                # {'milestones': [150, 300, 450], 'gamma': 1},
                # {'milestones': [150, 300, 450], 'gamma': 1},
                {'milestones': [1], 'gamma': 0.3},
                {'milestones': [1], 'gamma': 0.3},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.feeder_embedding.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v5',
                         'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': 10240,
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': None,
                         'level_coeff': 5,
                         'item': 'embedding',
                         },
                    'batch_size': 256,
                    'shuffle': True,
                    'num_workers': 8,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': None},
                {'weight': None},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_step': 1,
                      'train_state_save_step': 1,
                      }
        train_arg = {'epoch': 20480,
                     'eval_step': 1,
                     'phases': ['pre_gen', 'pre_dis', 'adv', ],
                     'phase_epochs': [0, 0, 20480],
                     'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 0,
                     'generator_pretrain_epoch': 0,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [1000],
                     'gen_lambda': [1, 1],
                     # 'noise_level_step': [30, 60, 90, 120],
                     # 'noise_level_step': [30],
                     'noise_level_step': [15, 30, 45, 60, 75],
                     'noise_level': [1, 0.8, 0.6, 0.4, 0.2, 0.],
                     # 'noise_level': [10, 10, 5, 5, 5],
                     # 'noise_level': [5, 2.5, 1, 0.5, 0.25],
                     # 'noise_level': [0.5, 0.3, 0.1, 0.05, 0.],
                     'period': 15,
                     'lambda_gp': 10,
                     'lambda_cls': 1.,
                     'gp_type': 'lp',
                     # 'last_epoch': 12,
                     # 'last_phase': 'adversarial',
                     # 'last_epoch': 3,
                     # 'last_phase': 'pretrain_discriminator',
                     'start_epoch': 0,
                     # 'last_phase': 'pretrain_generator',
                     'start_phase': 'adv',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,

                       'embedding_decoder': 'postprocess.embedding_decode.EmbeddingDecode',
                       'embedding_decoder_args': {
                           # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                           'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                           'beat_divisor': 8,
                           'subset_path': r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/counter.pkl',
                       },
                       'output_device': 1,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'log_item': 'embedding',
                       'train_type': 'acwganwb',
                       'num_classes': 5,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 5},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)


def train_acgan_embeddingv7_only_coord(setting_name='acgan_embeddingv7_only_coord'):
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

    for gen_lr, dis_lr in [[0.00003, 0.00003]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.acgan_embeddingv7_only_coord.Generator', 'nn.net.acgan_embeddingv7_only_coord.Discriminator'],
            'params': [
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'noise_dim': 128,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'label_preprocess_dim': 1,
                    'audio_preprocess_dim': 16,
                    'condition_coeff': 1.,
                    # star
                    'cls_label_dim': 1,
                },
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'preprocess_dim': 32,
                    # star
                    'cls_label_dim': 1,
                    'validity_sigmoid': False,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['RMSprop', 'RMSprop'],
            'params': [
                {'lr': gen_lr,},
                {'lr': dis_lr,},
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['MultiStepLR', 'MultiStepLR'],
            'params': [
                # {'milestones': [150, 300, 450], 'gamma': 1},
                # {'milestones': [150, 300, 450], 'gamma': 1},
                {'milestones': [1], 'gamma': 0.3},
                {'milestones': [1], 'gamma': 0.3},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.feeder_embedding.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v5',
                         'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': 10240,
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': None,
                         'level_coeff': 0.5,
                         'embedding_level_coeff': 0.5,
                         'item': 'coord',
                         },
                    'batch_size': 256,
                    'shuffle': True,
                    'num_workers': 4,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': None},
                {'weight': None},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_step': 1,
                      'train_state_save_step': 1,
                      }
        train_arg = {'epoch': 20480,
                     'eval_step': 1,
                     'phases': ['pre_gen', 'pre_dis', 'adv', ],
                     'phase_epochs': [0, 0, 20480],
                     'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 0,
                     'generator_pretrain_epoch': 0,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [1000],
                     'gen_lambda': [1, 1],
                     # 'noise_level_step': [30, 60, 90, 120],
                     # 'noise_level_step': [30],
                     # 'noise_level_step': [15, 30, 45, 60, 75],
                     # 'noise_level': [1, 0.8, 0.6, 0.4, 0.2, 0.],
                     # 'noise_level_step': [3, 6, 9, 12, 15],
                     # 'noise_level': [1, 0.8, 0.6, 0.4, 0.2, 0.],
                     # 'noise_level': [10, 10, 5, 5, 5],
                     # 'noise_level': [10, 10, 5, 5, 5],
                     # 'noise_level': [5, 2.5, 1, 0.5, 0.25],
                     # 'noise_level': [0.5, 0.3, 0.1, 0.05, 0.],
                     # 'noise_level_step': (np.arange(1, 6, 1) * 5).tolist(),
                     # 'noise_level': [0.5, 0.3, 0.1, 0.05, 0.025, 0.],
                     'noise_sched_type': 'util.train_util.BatchAbsCosineSchedulerMod',
                     'noise_sched_arg': {
                         'y_coeff': 0.5,
                         'x_coeff': 0.025,
                         'period': 32,
                     },
                     # 'noise_level': [0., 0.],
                     'lambda_gp': 10,
                     'lambda_cls': 10.,
                     'gp_type': 'lp',
                     # 'last_epoch': 12,
                     # 'last_phase': 'adversarial',
                     # 'last_epoch': 3,
                     # 'last_phase': 'pretrain_discriminator',
                     'start_epoch': 0,
                     # 'last_phase': 'pretrain_generator',
                     'start_phase': 'adv',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,

                       'embedding_decoder': 'postprocess.embedding_decode.EmbeddingDecode',
                       'embedding_decoder_args': {
                           # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                           'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                           'beat_divisor': 8,
                           'subset_path': r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/counter.pkl',
                       },
                       'output_device': 2,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'log_item': 'coord',
                       'train_type': 'acwganwb',
                       'num_classes': 5,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack',
                       # 'grad_alter_fn': None,
                       'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 1},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)


def train_acgan_embeddingv7_only_embedding(setting_name='acgan_embeddingv7_only_embedding'):
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

    for gen_lr, dis_lr in [[0.00003, 0.00003]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.acgan_embeddingv7_only_embedding.Generator', 'nn.net.acgan_embeddingv7_only_embedding.Discriminator'],
            'params': [
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'noise_dim': 128,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'label_preprocess_dim': 1,
                    'audio_preprocess_dim': 16,
                    'condition_coeff': 1.,
                    # star
                    'cls_label_dim': 1,
                },
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'preprocess_dim': 32,
                    # star
                    'cls_label_dim': 1,
                    'validity_sigmoid': False,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['RMSprop', 'RMSprop'],
            'params': [
                {'lr': gen_lr,},
                {'lr': dis_lr,},
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['MultiStepLR', 'MultiStepLR'],
            'params': [
                # {'milestones': [150, 300, 450], 'gamma': 1},
                # {'milestones': [150, 300, 450], 'gamma': 1},
                {'milestones': [1], 'gamma': 0.3},
                {'milestones': [1], 'gamma': 0.3},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.feeder_embedding.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v5',
                         'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': 10240,
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': None,
                         'level_coeff': 0.5,
                         'embedding_level_coeff': 0.5,
                         'item': 'embedding',
                         },
                    'batch_size': 256,
                    'shuffle': True,
                    'num_workers': 4,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': None},
                {'weight': None},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_step': 1,
                      'train_state_save_step': 1,
                      }
        train_arg = {'epoch': 20480,
                     'eval_step': 1,
                     'phases': ['pre_gen', 'pre_dis', 'adv', ],
                     'phase_epochs': [0, 0, 20480],
                     'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 0,
                     'generator_pretrain_epoch': 0,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [1000],
                     'gen_lambda': [1, 1],
                     # 'noise_level_step': [30, 60, 90, 120],
                     # 'noise_level_step': [30],
                     # 'noise_level_step': [3, 6, 9, 12, 15],
                     # 'noise_level': [1, 0.8, 0.6, 0.4, 0.2, 0.],
                     # 'noise_level': [10, 10, 5, 5, 5],
                     # 'noise_level': [5, 2.5, 1, 0.5, 0.25],
                     'noise_sched_type': 'util.train_util.BatchAbsCosineSchedulerMod',
                     'noise_sched_arg': {
                         'y_coeff': 0.5,
                         'x_coeff': 0.025,
                         'period': 32,
                     },
                     # 'noise_level_step': (np.arange(1, 6, 1) * 5).tolist(),
                     # 'noise_level': [0.5, 0.3, 0.1, 0.05, 0.025, 0.],
                     # 'noise_level': [0., 0.],
                     'lambda_gp': 10,
                     'lambda_cls': 10.,
                     'gp_type': 'lp',
                     # 'last_epoch': 12,
                     # 'last_phase': 'adversarial',
                     # 'last_epoch': 3,
                     # 'last_phase': 'pretrain_discriminator',
                     'start_epoch': 0,
                     # 'last_phase': 'pretrain_generator',
                     'start_phase': 'adv',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,

                       'embedding_decoder': 'postprocess.embedding_decode.EmbeddingDecode',
                       'embedding_decoder_args': {
                           # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                           'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                           'beat_divisor': 8,
                           'subset_path': r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/counter.pkl',
                       },
                       'output_device': 2,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'log_item': 'embedding',
                       'train_type': 'acwganwb',
                       'num_classes': 5,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack',
                       # 'grad_alter_fn': None,
                       'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 1},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)


def train_acgan_embeddingv7(setting_name='acgan_embeddingv7'):
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

    for gen_lr, dis_lr in [[0.0001, 0.00003]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.acgan_embeddingv7.Generator', 'nn.net.acgan_embeddingv7.Discriminator'],
            'params': [
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'noise_dim': 128,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'label_preprocess_dim': 1,
                    'audio_preprocess_dim': 24,
                    'condition_coeff': 1.,
                    # star
                    'cls_label_dim': 1,
                },
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'preprocess_dim': 32,
                    # star
                    'cls_label_dim': 1,
                    'validity_sigmoid': False,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['RMSprop', 'RMSprop'],
            'params': [
                {'lr': gen_lr,},
                {'lr': dis_lr,},
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['MultiStepLR', 'MultiStepLR'],
            'params': [
                # {'milestones': [150, 300, 450], 'gamma': 1},
                # {'milestones': [150, 300, 450], 'gamma': 1},
                {'milestones': [1], 'gamma': 0.3},
                {'milestones': [1], 'gamma': 0.3},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.feeder_embedding.SubseqFeeder',
                    'train_dataset_arg':
                        {
                            # 'save_dir': r'/home/data1/xiezheng/osu_mapper/preprocessed_v5',
                         'save_dir': r'/home/xiezheng/data/preprocessed_v5',
                         'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': None,
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': None,
                         'level_coeff': 0.5,
                         'embedding_level_coeff': 0.5,
                         'coord_level_coeff': None,
                         'item': 'coord_embedding',
                         },
                    'batch_size': 256,
                    'shuffle': True,
                    'num_workers': 8,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': None},
                {'weight': None},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_step': 1,
                      'train_state_save_step': 1,
                      }
        train_arg = {'epoch': 20480,
                     'eval_step': 1,
                     'phases': ['pre_gen', 'pre_dis', 'adv', ],
                     'phase_epochs': [0, 0, 20480],
                     'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 0,
                     'generator_pretrain_epoch': 0,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [1000],
                     'gen_lambda': [1, 1],
                     # 'noise_level_step': [30, 60, 90, 120],
                     # 'noise_level_step': [30],
                     # 'noise_level_step': [3, 6, 9, 12, 15],
                     # 'noise_level': [1, 0.8, 0.6, 0.4, 0.2, 0.],
                     # 'noise_level': [10, 10, 5, 5, 5],
                     # 'noise_level': [5, 2.5, 1, 0.5, 0.25],
                     'noise_sched_type': 'util.train_util.BatchAbsCosineSchedulerMod',
                     'noise_sched_arg': {
                         'y_coeff': 0.5,
                         'x_coeff': 0.025,
                         'low_bound': 0.3,
                         'period': 32,
                     },
                     # 'noise_level_step': (np.arange(1, 6, 1) * 5).tolist(),
                     # 'noise_level': [0.5, 0.3, 0.1, 0.05, 0.025, 0.],
                     # 'noise_level': [0., 0.],
                     'lambda_gp': 10,
                     'lambda_cls': 25.,
                     'gp_type': 'lp',
                     # 'last_epoch': 12,
                     # 'last_phase': 'adversarial',
                     # 'last_epoch': 3,
                     # 'last_phase': 'pretrain_discriminator',
                     'start_epoch': 0,
                     # 'last_phase': 'pretrain_generator',
                     'start_phase': 'adv',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,

                       'embedding_decoder': 'postprocess.embedding_decode.EmbeddingDecode',
                       'embedding_decoder_args': {
                           # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                           'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                           'beat_divisor': 8,
                           'subset_path': r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/counter.pkl',
                       },
                       'output_device': 2,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'log_item': 'coord_embedding',
                       'train_type': 'acwganwb',
                       'num_classes': 5,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack',
                       # 'grad_alter_fn': None,
                       'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 1},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)


def test_gen_capacity_v7(setting_name='test_gan_capacity'):
    """
    acgan_embeddingv5
    """
    random_seed = 404
    #
    # mel_args = {
    #     'sample_rate': 22000,
    #     'n_fft': 512,
    #     'hop_length': 220,  # 10 ms
    #     'n_mels': 40,
    # }
    # epoch = 20480

    # 32 beat per subseq
    beat_divisor = 8
    subseq_beats = 32
    subseq_snaps = subseq_beats * beat_divisor
    embedding_size = subseq_beats * 16
    label_size = 2 * subseq_snaps
    rnd_bank_size = max(embedding_size, label_size) * 1024
    mel_features = 40

    for gen_lr, dis_lr in [[0.00001, 0.00001]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': 'nn.net.acgan_embeddingv5.Generator',

            'n_snap': subseq_snaps,
            'tgt_embedding_dim': 16,
            'tgt_coord_dim': 2,
            'audio_feature_dim': mel_features,
            'noise_dim': 128,
            'norm': 'LN',
            'middle_dim': 128,
            'label_preprocess_dim': 1,
            'audio_preprocess_dim': 4,
            'condition_coeff': 1.,
            # star
            'cls_label_dim': 1,
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': 'RMSprop',
            'lr': gen_lr,
        }
        scheduler_arg = {
            'scheduler_type': 'MultiStepLR',
            'milestones': [50],
            'gamma': 1.,
        }
        data_arg = {'dataset': 'nn.dataset.feeder_embedding.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v5',
                         'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': 1,
                         'take_first_subseq': 1,
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': rnd_bank_size,
                         'level_coeff': 0.5,
                         },
                    'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {
                       'loss_type': 'MSELoss',
                   }
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_step': 1,
                      'train_state_save_step': 1,
                      }
        train_arg = {'epoch': 20480,
                     'eval_step': 1,
                     'phases': ['pre_gen', 'pre_dis', 'adv', ],
                     'phase_epochs': [0, 0, 20480],
                     'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 0,
                     'generator_pretrain_epoch': 0,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 2,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [1000],
                     'gen_lambda': [1, 1],
                     # 'noise_level_step': [(50 + 25 * i) for i in range(9)],
                     # 'noise_level': [(1 - 0.1 * i) for i in range(10)],
                     'noise_level_step': [30, 60, 90, 120],
                     'noise_level': [5, 2.5, 1, 0.5],
                     'period': 15,
                     # 'noise_level': [0.5, 0.3, 0.1, 0.05, 0.],
                     'lambda_gp': 10,
                     'lambda_cls': 10.,
                     # 'last_epoch': 12,
                     # 'last_phase': 'adversarial',
                     # 'last_epoch': 3,
                     # 'last_phase': 'pretrain_discriminator',
                     'start_epoch': 0,
                     # 'last_phase': 'pretrain_generator',
                     'start_phase': 'adv',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,

                       'embedding_decoder': 'postprocess.embedding_decode.EmbeddingDecode',
                       'embedding_decoder_args': {
                           'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                           'beat_divisor': 8,
                           'subset_path': r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/counter.pkl',
                       },
                       'output_device': 7,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'regression',
                       'num_classes': 5,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack',
                       # 'grad_alter_fn': None,
                       'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 5},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)


def train_acgan_embeddingv8(setting_name='acgan_embeddingv8'):
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

    for gen_lr, dis_lr in [[0.000003, 0.00001]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.acgan_embeddingv8.Generator', 'nn.net.acgan_embeddingv8.Discriminator'],
            'params': [
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'noise_dim': 128,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'label_preprocess_dim': 1,
                    'audio_preprocess_dim': 24,
                    'condition_coeff': 1.,
                    # star
                    'cls_label_dim': 1,
                },
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'audio_preprocess_dim': 24,
                    # star
                    'cls_label_dim': 1,
                    'validity_sigmoid': False,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['RMSprop', 'RMSprop'],
            'params': [
                {'lr': gen_lr,},
                {'lr': dis_lr,},
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['MultiStepLR', 'MultiStepLR'],
            'params': [
                # {'milestones': [150, 300, 450], 'gamma': 1},
                # {'milestones': [150, 300, 450], 'gamma': 1},
                {'milestones': [1], 'gamma': 1},
                {'milestones': [1], 'gamma': 1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.feeder_embedding.SubseqFeeder',
                    'train_dataset_arg':
                        {
                            # 'save_dir': r'/home/data1/xiezheng/osu_mapper/preprocessed_v5',
                         'save_dir': r'/home/xiezheng/data/preprocessed_v5',
                         'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': None,
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': None,
                         'level_coeff': 0.5,
                         'embedding_level_coeff': 0.5,
                         'coord_level_coeff': 0,
                         'item': 'coord_embedding',
                         },
                    'batch_size': 256,
                    'shuffle': True,
                    'num_workers': 16,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': None},
                {'weight': None},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_step': 1,
                      'model_save_batch_step': 250,
                      'log_batch_step': 250,
                      'train_state_save_step': 1,
                      }
        train_arg = {'epoch': 20480,
                     'eval_step': 1,
                     'phases': ['pre_gen', 'pre_dis', 'adv', ],
                     'phase_epochs': [0, 0, 20480],
                     'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 0,
                     'generator_pretrain_epoch': 0,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [1000],
                     'gen_lambda': [1, 1],
                     # 'noise_level_step': [30, 60, 90, 120],
                     # 'noise_level_step': [30],
                     # 'noise_level_step': [3, 6, 9, 12, 15],
                     # 'noise_level': [1, 0.8, 0.6, 0.4, 0.2, 0.],
                     # 'noise_level': [10, 10, 5, 5, 5],
                     # 'noise_level': [5, 2.5, 1, 0.5, 0.25],
                     'noise_sched_type': 'util.train_util.BatchAbsCosineSchedulerMod',
                     'noise_sched_arg': {
                         'y_coeff': 0.5,
                         'x_coeff': 0.025,
                         'low_bound': 0.3,
                         'period': 32,
                     },
                     # 'noise_level_step': (np.arange(1, 6, 1) * 5).tolist(),
                     # 'noise_level': [0.5, 0.3, 0.1, 0.05, 0.025, 0.],
                     # 'noise_level': [0., 0.],
                     'lambda_gp': 10,
                     'lambda_cls': 5.,
                     'gp_type': 'lp',
                     # 'last_epoch': 12,
                     # 'last_phase': 'adversarial',
                     # 'last_epoch': 3,
                     # 'last_phase': 'pretrain_discriminator',
                     'start_epoch': 0,
                     # 'last_phase': 'pretrain_generator',
                     'start_phase': 'adv',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,

                       'embedding_decoder': 'postprocess.embedding_decode.EmbeddingDecode',
                       'embedding_decoder_args': {
                           # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                           'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                           'beat_divisor': 8,
                           'subset_path': r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/counter.pkl',
                       },
                       'output_device': 3,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'log_item': 'coord_embedding',
                       'train_type': 'acwganwb',
                       'num_classes': 5,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack',
                       # 'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 1},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)


def train_acgan_embeddingv9(setting_name='acgan_embeddingv9'):
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

    for gen_lr, dis_lr in [[1e-4, 1e-4]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.acgan_embeddingv9.Generator', 'nn.net.acgan_embeddingv9.Discriminator'],
            'params': [
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'noise_dim': 128,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'label_preprocess_dim': 1,
                    'audio_preprocess_dim': 24,
                    'condition_coeff': 1.,
                    # star
                    'cls_label_dim': 1,
                },
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'audio_preprocess_dim': 24,
                    # star
                    'cls_label_dim': 1,
                    'validity_sigmoid': False,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['RMSprop', 'RMSprop'],
            'params': [
                {'lr': gen_lr,},
                {'lr': dis_lr,},
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['MultiStepLR', 'MultiStepLR'],
            'params': [
                # {'milestones': [150, 300, 450], 'gamma': 1},
                # {'milestones': [150, 300, 450], 'gamma': 1},
                {'milestones': [1], 'gamma': 1},
                {'milestones': [1], 'gamma': 1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.feeder_embedding.SubseqFeeder',
                    'train_dataset_arg':
                        {
                            # 'save_dir': r'/home/data1/xiezheng/osu_mapper/preprocessed_v5',
                         'save_dir': r'/home/xiezheng/data/preprocessed_v5',
                         'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': None,
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': None,
                         'level_coeff': 0.5,
                         'embedding_level_coeff': 0.5,
                         'coord_level_coeff': 0.5,
                         'item': 'coord_embedding',
                         },
                    'batch_size': 256,
                    'shuffle': True,
                    'num_workers': 16,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': None},
                {'weight': None},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_step': 1,
                      'model_save_batch_step': 250,
                      'log_batch_step': 250,
                      'train_state_save_step': 1,
                      }
        train_arg = {'epoch': 20480,
                     'eval_step': 1,
                     'phases': ['pre_gen', 'pre_dis', 'adv', ],
                     'phase_epochs': [0, 0, 20480],
                     'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 0,
                     'generator_pretrain_epoch': 0,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [1000],
                     'gen_lambda': [1, 1],
                     # 'noise_level_step': [30, 60, 90, 120],
                     # 'noise_level_step': [30],
                     # 'noise_level_step': [3, 6, 9, 12, 15],
                     # 'noise_level': [1, 0.8, 0.6, 0.4, 0.2, 0.],
                     # 'noise_level': [10, 10, 5, 5, 5],
                     # 'noise_level': [5, 2.5, 1, 0.5, 0.25],
                     'noise_sched_type': 'util.train_util.BatchAbsCosineSchedulerMod',
                     'noise_sched_arg': {
                         'y_coeff': 0.5,
                         'x_coeff': 0.025,
                         'low_bound': 0.3,
                         'period': None,
                     },
                     # 'noise_level_step': (np.arange(1, 6, 1) * 5).tolist(),
                     # 'noise_level': [0.5, 0.3, 0.1, 0.05, 0.025, 0.],
                     # 'noise_level': [0., 0.],
                     'lambda_gp': 10,
                     'lambda_cls': 1.,
                     'gp_type': 'lp',
                     # 'last_epoch': 12,
                     # 'last_phase': 'adversarial',
                     # 'last_epoch': 3,
                     # 'last_phase': 'pretrain_discriminator',
                     'start_epoch': 0,
                     # 'last_phase': 'pretrain_generator',
                     'start_phase': 'adv',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,

                       'embedding_decoder': 'postprocess.embedding_decode.EmbeddingDecode',
                       'embedding_decoder_args': {
                           # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                           'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                           'beat_divisor': 8,
                           'subset_path': r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/counter.pkl',
                       },
                       'output_device': 3,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'log_item': 'coord_embedding',
                       'train_type': 'acwganwb',
                       'num_classes': 5,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack',
                       # 'grad_alter_fn': None,
                       'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 5},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)

