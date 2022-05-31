import yaml

from nn.train import train_with_config


def train_mel_mlp_density_c1(setting_name):
    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'net.mel_mlp_reg_c1.MelMLPRegC1',
                     'extract_hidden_layer_num': 1,
                     'snap_mel': 4,
                     'n_mel': 128,
                     'sample_beats': 8,
                     'pad_beats': 2,
                     'snap_divisor': 8,
                     }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 2, 'gamma': 0.1}
        data_arg = {'dataset': 'dataset.mel_db_dataset_c1.MelDBDatasetC1',
                    'train_dataset_arg':
                        {'db_path': r'./resources/data/osu_train_mel.db',
                         'audio_dir': r'./resources/data/mel',
                         'table_name': 'TRAINFOLD%d',
                         'snap_mel': 4,
                         'snap_offset': 0,
                         'snap_divisor': 8,
                         'sample_beats': 8,
                         'pad_beats': 2,
                         'multi_label': True,
                         'density_label': True, },
                    'test_dataset_arg':
                        {'db_path': r'./resources/data/osu_train_mel.db',
                         'audio_dir': r'./resources/data/mel',
                         'table_name': 'TESTFOLD%d',
                         'snap_mel': 4,
                         'snap_offset': 0,
                         'snap_divisor': 8,
                         'sample_beats': 8,
                         'pad_beats': 2,
                         'multi_label': True,
                         'density_label': True, },
                    'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': 'loss.multi_reg_loss.MultiMSELoss'}
        pred_arg = {'pred_type': 'pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 4}
        train_arg = {'epoch': 16, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'task_type': 'regression',
                       'collate_fn': 'dataset.collate_fn.data_array_to_tensor',
                       'cal_acc_func': 'metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=5)


def train_mel_mlp(setting_name):
    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'net.mel_mlp.MelMLP',
                     'num_classes': 3,
                     'extract_hidden_layer_num': 2,
                     'snap_mel': 4,
                     'n_mel': 128,
                     'sample_beats': 16,
                     'pad_beats': 4,
                     'snap_divisor': 8,
                     }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 6, 'gamma': 0.1}
        data_arg = {'dataset': 'dataset.mel_db_dataset.MelDBDataset',
                    'train_dataset_arg':
                        {'db_path': r'./resources/data/osu_train_mel.db',
                         'audio_dir': r'./resources/data/mel',
                         'table_name': 'TRAINFOLD%d',
                         'snap_mel': 4,
                         'snap_offset': 0,
                         'snap_divisor': 8,
                         'sample_beats': 16,
                         'pad_beats': 4,
                         'multi_label': True, },
                    'test_dataset_arg':
                        {'db_path': r'./resources/data/osu_train_mel.db',
                         'audio_dir': r'./resources/data/mel',
                         'table_name': 'TESTFOLD%d',
                         'snap_mel': 4,
                         'snap_offset': 0,
                         'snap_divisor': 8,
                         'sample_beats': 16,
                         'pad_beats': 4,
                         'multi_label': True, },
                    'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': 'loss.multi_pred_loss.MultiPredLoss'}
        pred_arg = {'pred_type': 'pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 1}
        train_arg = {'epoch': 24, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'collate_fn': 'dataset.collate_fn.data_array_to_tensor',
                       'cal_acc_func': 'metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=5)


def train_seg_multi_pred_mlp(setting_name):
    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'net.seg_multi_pred_mlp.SegMultiPredMLP',
                     'num_classes': 2,
                     'extract_hidden_layer_num': 0,
                     'beat_feature_frames': 16384,
                     'sample_beats': 16,
                     'pad_beats': 4,
                     }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 3, 'gamma': 0.1}
        data_arg = {'dataset': 'dataset.seg_multi_label_db_dataset.SegMultiLabelDBDataset',
                    'train_dataset_arg':
                        {'db_path': r'./resources/data/osu_train.db',
                         'audio_dir': r'./resources/data/audio',
                         'table_name': 'TRAINFOLD%d',
                         'beat_feature_frames': 16384,
                         # this 8+32+8 will yield an audio fragment of about 17 sec
                         'sample_beats': 16,
                         'pad_beats': 4,
                         'multi_label': False, },
                    'test_dataset_arg':
                        {'db_path': r'./resources/data/osu_train.db',
                         'audio_dir': r'./resources/data/audio',
                         'table_name': 'TESTFOLD%d',
                         'beat_feature_frames': 16384,
                         'sample_beats': 16,
                         'pad_beats': 4,
                         'multi_label': False, },
                    'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': 'loss.multi_pred_loss.MultiPredLoss'}
        pred_arg = {'pred_type': 'pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 1}
        train_arg = {'epoch': 24, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'collate_fn': 'dataset.collate_fn.data_array_to_tensor',
                       'cal_acc_func': 'metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=5)


def train_seg_multi_pred_cnn(setting_name):
    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'net.seg_multi_pred_cnn.SegMultiPredCNN',
                     'num_classes': 2,
                     'out_snaps': 128*8,
                     }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 1, 'gamma': 0.1}
        data_arg = {'dataset': 'dataset.seg_multi_label_db_dataset.SegMultiLabelDBDataset',
                    'train_dataset_arg':
                        {'db_path': r'./resources/data/osu_train.db',
                         'audio_dir': r'./resources/data/audio',
                         'table_name': 'TRAINFOLD%d',
                         'beat_feature_frames': 16384,
                         # this 8+32+8 will yield an audio fragment of about 17 sec
                         'sample_beats': 128,
                         'pad_beats': 32,
                         'multi_label': False, },
                    'test_dataset_arg':
                        {'db_path': r'./resources/data/osu_train.db',
                         'audio_dir': r'./resources/data/audio',
                         'table_name': 'TESTFOLD%d',
                         'beat_feature_frames': 16384,
                         'sample_beats': 128,
                         'pad_beats': 32,
                         'multi_label': False, },
                    'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': 'loss.multi_pred_loss.MultiPredLoss'}
        pred_arg = {'pred_type': 'pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 1}
        train_arg = {'epoch': 4, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'collate_fn': 'dataset.collate_fn.data_array_to_tensor',
                       'cal_acc_func': 'metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=5)


def train_rnnv1(setting_name):
    batch_size = 1
    subseq_len = 64
    random_seed = 404
    for lr in [0.01]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'nn.net.rnnv1.RNNv1',
                     'num_classes': 3,
                     'num_layers': 2,
                     'in_channels': 515,
                     }
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 8, 'gamma': 0.1}
        data_arg = {'dataset': 'nn.dataset.rnn_feeder.RNNFeeder',




                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnn/snap_1/train%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnn/snap_1/train%d_label.pkl',
                         'subseq_len': subseq_len,
                         'batch_size': batch_size,
                         'random_seed': random_seed,
                         'binary': False,
                         },
                    'test_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnn/snap_1/test%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnn/snap_1/test%d_label.pkl',
                         'subseq_len': subseq_len,
                         'batch_size': 1,
                         'random_seed': random_seed,
                         'binary': False,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': batch_size,
                    'drop_last': False}
        loss_arg = {'loss_type': 'CrossEntropy'}
        pred_arg = {'pred_type': None}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': 64, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'rnn': True,
                       'collate_fn': 'nn.dataset.collate_fn.data_array_to_tensor',
                       # 'cal_acc_func': 'metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       # 'cal_cm_func': 'metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn_arg': {'theta': 1},
                       }, f)
        train_with_config(config_path, format_config=True, folds=5)


def train_rnnv1_nolabel(setting_name):
    batch_size = 1
    subseq_len = 64
    random_seed = 404
    for lr in [0.01]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'nn.net.rnnv1.RNNv1',
                     'num_classes': 3,
                     'num_layers': 2,
                     'in_channels': 512,
                     }
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 2, 'gamma': 0.1}
        data_arg = {'dataset': 'nn.dataset.rnn_feeder.RNNFeeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnn_nolabel/train%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnn_nolabel/train%d_label.pkl',
                         'subseq_len': subseq_len,
                         'batch_size': batch_size,
                         'random_seed': random_seed,
                         'binary': False,
                         },
                    'test_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnn_nolabel/test%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnn_nolabel/test%d_label.pkl',
                         'subseq_len': subseq_len,
                         'batch_size': 1,
                         'random_seed': random_seed,
                         'binary': False,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': batch_size,
                    'drop_last': False}
        loss_arg = {'loss_type': 'CrossEntropy'}
        pred_arg = {'pred_type': None}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 4}
        train_arg = {'epoch': 8, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'rnn': True,
                       'collate_fn': 'nn.dataset.collate_fn.data_array_to_tensor',
                       # 'cal_acc_func': 'metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       # 'cal_cm_func': 'metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn_arg': {'theta': 1},
                       }, f)
        train_with_config(config_path, format_config=True, folds=1)


def train_rnnv1_bi(setting_name):
    batch_size = 1
    subseq_len = 1024
    random_seed = 404
    for lr in [0.01]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'nn.net.rnnv1.RNNv1',
                     'num_classes': 2,
                     'num_layers': 2,
                     'snaps': 1,
                     'snap_mel': 4,
                     'n_mel': 128,
                     }
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 4, 'gamma': 0.1}
        data_arg = {'dataset': 'nn.dataset.rnn_feeder.RNNFeeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnn/snap_1/train%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnn/snap_1/train%d_label.pkl',
                         'subseq_len': subseq_len,
                         'batch_size': batch_size,
                         'random_seed': random_seed,
                         'binary': True,},
                    'test_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnn/snap_1/test%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnn/snap_1/test%d_label.pkl',
                         'subseq_len': subseq_len,
                         'batch_size': 1,
                         'random_seed': random_seed,
                         'binary': True, },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': batch_size,
                    'drop_last': False}
        loss_arg = {'loss_type': 'CrossEntropy'}
        pred_arg = {'pred_type': None}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': 16, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'rnn': True,
                       'collate_fn': 'nn.dataset.collate_fn.data_array_to_tensor',
                       # 'cal_acc_func': 'metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       # 'cal_cm_func': 'metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn_arg': {'theta': 1},
                       }, f)
        train_with_config(config_path, format_config=True, folds=5)


def train_mlpv2(setting_name):
    batch_size = 64
    random_seed = 404
    mel_feature = 512
    meta_feature = 2
    label_feature = 16 * 3
    total_feature = mel_feature + label_feature + meta_feature
    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'nn.net.mlpv2.MLPv2',
                     'layers': [total_feature, total_feature*2, total_feature*2, 3],
                     }
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 2, 'gamma': 0.1}
        data_arg = {'dataset': 'nn.dataset.feeder.Feeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/mlp/snap_1/train%d_data.pkl',
                         'label_path': r'./resources/data/fit/mlp/snap_1/train%d_label.pkl',
                         },
                    'test_dataset_arg':
                        {'data_path': r'./resources/data/fit/mlp/snap_1/test%d_data.pkl',
                         'label_path': r'./resources/data/fit/mlp/snap_1/test%d_label.pkl',
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': 'CrossEntropy'}
        pred_arg = {'pred_type': None}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': 8, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'rnn': False,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       # 'cal_acc_func': 'metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       # 'cal_cm_func': 'metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       # 'grad_alter_fn': 'util.net_util.grad_clipping',
                       # 'grad_alter_fn_arg': {'theta': 1},
                       }, f)
        train_with_config(config_path, format_config=True, folds=1)


def train_mlpv2_nolabel(setting_name):
    batch_size = 64
    random_seed = 404
    mel_feature = 512
    meta_feature = 2
    label_feature = 0
    total_feature = mel_feature + label_feature + meta_feature
    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'nn.net.mlpv2.MLPv2',
                     'layers': [total_feature, total_feature*2, total_feature*2, 3],
                     }
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 2, 'gamma': 0.1}
        data_arg = {'dataset': 'nn.dataset.feeder.Feeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/mlp_nolabel/train%d_data.pkl',
                         'label_path': r'./resources/data/fit/mlp_nolabel/train%d_label.pkl',
                         },
                    'test_dataset_arg':
                        {'data_path': r'./resources/data/fit/mlp_nolabel/test%d_data.pkl',
                         'label_path': r'./resources/data/fit/mlp_nolabel/test%d_label.pkl',
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': 'CrossEntropy'}
        pred_arg = {'pred_type': None}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': 8, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'rnn': False,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       # 'cal_acc_func': 'metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       # 'cal_cm_func': 'metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       # 'grad_alter_fn': 'util.net_util.grad_clipping',
                       # 'grad_alter_fn_arg': {'theta': 1},
                       }, f)
        train_with_config(config_path, format_config=True, folds=1)


if __name__ == '__main__':
    # setting_name = 'mlpv2_lr0.1'
    # train_mlpv2(setting_name)
    setting_name = 'rnnv1_lr0.01'
    train_rnnv1(setting_name)
    # setting_name = 'mlpv2_nolabel_lr0.1'
    # train_mlpv2_nolabel(setting_name)

