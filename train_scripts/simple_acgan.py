import yaml

from nn.train import train_with_config


def train_simple_acganv3_heatmapv3(setting_name='simple_acganv3_heatmapv3'):
    """
    simple_acganv2
    no sigmoid
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

    # 25.6 s per subseq
    subseq_len = 2560
    snap_feature = 41

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.000005, 0.000005]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.simple_acganv3.Generator', 'nn.net.simple_acganv3.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'tgt_dim': 5,
                    'noise_dim': 16,
                    'audio_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 128,
                    'preprocess_dim': 16,
                    'cls_label_dim': 4,
                },
                {
                    'seq_len': subseq_len,
                    'tgt_dim': 5,
                    # 'noise_dim': 64,
                    'audio_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 128,
                    'preprocess_dim': 16,
                    'cls_label_dim': 4,
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
                {'milestones': [50], 'gamma': 1.},
                {'milestones': [50], 'gamma': 1.},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.feeder_heatmapv3_meta.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'/home/data1/xiezheng/osu_mapper/preprocessed',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': 20480,
                         'pad': False,
                         },
                    'batch_size': 256 * 8,
                    'shuffle': False,
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
                      'model_save_step': 8,
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
                     # 'noise_level_step': [(50 + 25 * i) for i in range(9)],
                     # 'noise_level': [(1 - 0.1 * i) for i in range(10)],
                     # 'noise_level_step': [30, 60, 90, 120],
                     # 'noise_level': [0.5, 0.5, 0.5, 0.5, 0.5],
                     'noise_level_step': [30],
                     'noise_level': [1., 1.],
                     # 'noise_level': [0.5, 0.3, 0.1, 0.05, 0.],
                     'lambda_gp': 10,
                     'lambda_cls': 5.,
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
                       'output_device': 0,
                       'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'acwganwb',
                       'num_classes': 5,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)


if __name__ == '__main__':
    # train_simple_acganv1_heatmapv2('simple_acganv1_heatmapv2_20230818_g0.0001_d0.0001')
    # train_simple_acganv1_heatmapv2('simple_acganv1_heatmapv2_20230821_g0.00003_d0.00003_rev1')
    # train_simple_acganv2_heatmapv2('simple_acganv2_heatmapv2_20230823_g0.0001_d0.0001_constlr')
    # train_simple_acganv3_heatmapv3('simple_acganv3_heatmapv3_20230829_g0.00001_d0.00001_constlr')
    train_simple_acganv3_heatmapv3('simple_acganv3_heatmapv3_20230830_g0.000005_d0.000005_constlr')