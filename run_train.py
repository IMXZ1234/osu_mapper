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
    for lr in [0.01]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'nn.net.mel_mlp.MelMLP',
                     'num_classes': 3,
                     'extract_hidden_layer_num': 2,
                     'snap_mel': 4,
                     'n_mel': 128,
                     'sample_beats': 16,
                     'pad_beats': 4,
                     'snap_divisor': 8,
                     }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 1, 'gamma': 0.1}
        data_arg = {'dataset': 'nn.dataset.mel_db_dataset.MelDBDataset',
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
        loss_arg = {'loss_type': 'nn.loss.multi_pred_loss.MultiPredLoss'}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 1}
        train_arg = {'epoch': 4, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'collate_fn': 'nn.dataset.collate_fn.data_array_to_tensor',
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1)


def train_mel_mlp_c1norm(setting_name):
    switch_label = True
    if switch_label:
        num_classes = 4
        weights = [1, 10, 10, 10]
    else:
        num_classes = 3
        weights = None
    for lr in [0.01]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'nn.net.mel_mlp_c1.MelMLPC1',
                     'num_classes': num_classes,
                     'extract_hidden_layer_num': 2,
                     'snap_mel': 4,
                     'n_mel': 128,
                     'sample_beats': 16,
                     'pad_beats': 4,
                     'snap_divisor': 8,
                     }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 1, 'gamma': 0.1}
        data_arg = {'dataset': 'nn.dataset.mel_db_dataset_c1norm.MelDBDatasetC1Norm',
                    'train_dataset_arg':
                        {'db_path': r'./resources/data/osu_train_mel.db',
                         'audio_dir': r'./resources/data/mel',
                         'table_name': 'TRAINFOLD%d',
                         'snap_mel': 4,
                         'snap_offset': 0,
                         'snap_divisor': 8,
                         'sample_beats': 16,
                         'pad_beats': 4,
                         'switch_label': switch_label,
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
                         'switch_label': switch_label,
                         'multi_label': True, },
                    'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': 'nn.loss.multi_pred_loss.MultiPredLoss', 'weights': weights}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 1}
        train_arg = {'epoch': 4, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'collate_fn': 'nn.dataset.collate_fn.data_array_to_tensor',
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


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
                     'out_snaps': 128 * 8,
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
        train_with_config(config_path, format_config=True, folds=1)


def train_rnnv2(setting_name):
    batch_size = 4
    subseq_len = 64
    random_seed = 404
    for lr in [0.01]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'nn.net.rnnv2.RNNv2',
                     'num_classes': 3,
                     'num_layers': 1,
                     'in_channels': 517,
                     }
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 1, 'gamma': 0.1}
        data_arg = {'dataset': 'nn.dataset.rnnv2_feeder.RNNv2Feeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnn_v2/train%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnn_v2/train%d_label.pkl',
                         'subseq_len': subseq_len,
                         'batch_size': batch_size,
                         'random_seed': random_seed,
                         'binary': False,
                         },
                    'test_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnn_v2/test%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnn_v2/test%d_label.pkl',
                         'subseq_len': subseq_len,
                         'batch_size': batch_size,
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
        train_arg = {'epoch': 4, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'rnn': True,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate_other',
                       # 'cal_acc_func': 'metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       # 'cal_cm_func': 'metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn_arg': {'theta': 1},
                       }, f)
        train_with_config(config_path, format_config=True, folds=1)


def train_rnnv3(setting_name):
    batch_size = 1
    subseq_len = 64
    random_seed = 404
    for lr in [0.01]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'nn.net.rnnv2.RNNv2',
                     'num_classes': 4,
                     'num_layers': 2,
                     'in_channels': 518,
                     }
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 2, 'gamma': 0.1}
        data_arg = {'dataset': 'nn.dataset.rnnv2_feeder.RNNv2Feeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnnv3/train%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnnv3/train%d_label.pkl',
                         'subseq_len': subseq_len,
                         'batch_size': batch_size,
                         'random_seed': random_seed,
                         'binary': False,
                         },
                    'test_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnnv3/test%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnnv3/test%d_label.pkl',
                         'subseq_len': subseq_len,
                         'batch_size': batch_size,
                         'random_seed': random_seed,
                         'binary': False,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': batch_size,
                    'drop_last': False}
        loss_arg = {'loss_type': 'CrossEntropy', 'weight': [1., 10., 10., 10.]}
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
                       'rnn': True,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate_other',
                       # 'cal_acc_func': 'metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       # 'cal_cm_func': 'metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn_arg': {'theta': 1},
                       }, f)
        train_with_config(config_path, format_config=True, folds=1)


def train_rnnv3_nolabel(setting_name):
    batch_size = 16
    subseq_len = 64
    random_seed = 404
    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'nn.net.rnnv2.RNNv2',
                     'num_classes': 4,
                     'num_layers': 4,
                     'in_channels': 514,
                     }
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 8, 'gamma': 0.1}
        data_arg = {'dataset': 'nn.dataset.rnnv2_feeder.RNNv2Feeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnnv3_nolabel/train%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnnv3_nolabel/train%d_label.pkl',
                         'subseq_len': subseq_len,
                         'batch_size': batch_size,
                         'random_seed': random_seed,
                         'binary': False,
                         },
                    'test_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnnv3_nolabel/test%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnnv3_nolabel/test%d_label.pkl',
                         'subseq_len': subseq_len,
                         'batch_size': batch_size,
                         'random_seed': random_seed,
                         'binary': False,
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
        train_arg = {'epoch': 32, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'rnn': True,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate_other',
                       # 'cal_acc_func': 'metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       # 'cal_cm_func': 'metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn_arg': {'theta': 1},
                       }, f)
        train_with_config(config_path, format_config=True, folds=1)


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
                         'binary': True, },
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
                     'layers': [total_feature, total_feature * 2, total_feature * 2, 3],
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
                     'layers': [total_feature, total_feature * 2, total_feature * 2, 3],
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


def train_cganv1(setting_name):
    switch_label = False
    if switch_label:
        num_classes = 4
        weights = [1, 10, 10, 10]
    else:
        num_classes = 3
        weights = None
    scheduler_step_size = 1

    extract_hidden_layer_num = 2
    snap_mel = 4
    n_mel = 128
    sample_beats = 16
    pad_beats = 4
    snap_divisor = 8

    noise_feature_num = sample_beats * snap_divisor
    meta_feature_num = 2  # bpm, speed_stars

    sample_padded_beats = sample_beats + pad_beats * 2

    for lr in [0.01]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv1.Generator', 'nn.net.cganv1.Discriminator'],
            'params': [
                {
                    'data_feature_num': n_mel * snap_mel * sample_padded_beats + meta_feature_num,
                    'noise_feature_num': noise_feature_num,
                    'output_feature_num': sample_beats * snap_divisor,
                },
                {
                    'data_feature_num': n_mel * snap_mel * sample_padded_beats + meta_feature_num,
                    'output_feature_num': sample_beats * snap_divisor,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['SGD', 'SGD'],
            'params': [
                {'lr': lr},
                {'lr': lr},
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['StepLR', 'StepLR'],
            'params': [
                {'step_size': scheduler_step_size, 'gamma': 0.1},
                {'step_size': scheduler_step_size, 'gamma': 0.1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.mel_db_dataset_c1norm.MelDBDatasetC1Norm',
                    'train_dataset_arg':
                        {'db_path': r'./resources/data/osu_train_mel.db',
                         'audio_dir': r'./resources/data/mel',
                         'table_name': 'TRAINFOLD%d',
                         'snap_mel': 4,
                         'snap_offset': 0,
                         'snap_divisor': 8,
                         'sample_beats': 16,
                         'pad_beats': 4,
                         'switch_label': switch_label,
                         'multi_label': True,
                         'flatten_feature': True},
                    'test_dataset_arg':
                        {'db_path': r'./resources/data/osu_train_mel.db',
                         'audio_dir': r'./resources/data/mel',
                         'table_name': 'TESTFOLD%d',
                         'snap_mel': 4,
                         'snap_offset': 0,
                         'snap_divisor': 8,
                         'sample_beats': 16,
                         'pad_beats': 4,
                         'switch_label': switch_label,
                         'multi_label': True,
                         'flatten_feature': True},
                    'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            'nn.loss.multi_pred_loss.MultiPredLoss',
            'CrossEntropy',
        ],
            'params': [
                {'weights': weights},
                {'weights': weights},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 1}
        train_arg = {'epoch': 4, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'collate_fn': 'nn.dataset.collate_fn.data_array_to_tensor',
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_rnnv4(setting_name):
    batch_size = 1
    subseq_len = 1
    random_seed = 404

    step_snaps = 6 * 8
    for lr in [0.01]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'nn.net.rnnv2_seg.RNNv2Seg',
                     'num_classes': 3,
                     'num_layers': 2,
                     'in_channels': 24722,
                     'step_snaps': step_snaps,
                     }
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 32, 'gamma': 0.1}
        data_arg = {'dataset': 'nn.dataset.rnnv2_feeder.RNNv2Feeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnnv4/train%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnnv4/train%d_label.pkl',
                         'subseq_len': subseq_len,
                         'batch_size': batch_size,
                         'random_seed': random_seed,
                         'binary': False,
                         },
                    'test_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnnv4/test%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnnv4/test%d_label.pkl',
                         'subseq_len': subseq_len,
                         'batch_size': batch_size,
                         'random_seed': random_seed,
                         'binary': False,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': batch_size,
                    'drop_last': False}
        loss_arg = {'loss_type': 'nn.loss.multi_pred_loss.MultiPredLoss', 'weight': [1., 10., 1.]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 32}
        train_arg = {'epoch': 128, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'rnn',
                       'collate_fn': 'nn.dataset.collate_fn.default_collate_other',
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn_arg': {'theta': 1},
                       }, f)
        train_with_config(config_path, format_config=True, folds=1)


def train_seq2seq(setting_name):
    batch_size = 4
    # 12 beats
    subseq_len = 12 * 8
    random_seed = 404

    step_snaps = 6 * 8
    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {
            'model_type': 'nn.net.seq2seq.Seq2Seq',
            'encoder_args': {
                'input_size': 514,
                'embed_size': 512,
                'hidden_size': 2048,
                'n_layers': 2,
                'dropout': 0.5,
            },
            'decoder_args': {
                'embed_size': 3,
                'hidden_size': 2048,
                'output_size': 3,
                'n_layers': 1,
                'dropout': 0.5,
            },
        }
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 32, 'gamma': 0.1}
        # data_arg = {'dataset': 'nn.dataset.feeder.Feeder',
        #             'train_dataset_arg':
        #                 {'data_path': r'./resources/data/fit/rnnv3_nolabel/train%d_data.pkl',
        #                  'label_path': r'./resources/data/fit/rnnv3_nolabel/train%d_label.pkl',
        #                  # 'subseq_len': subseq_len,
        #                  # 'batch_size': batch_size,
        #                  # 'random_seed': random_seed,
        #                  # 'binary': False,
        #                  },
        #             'test_dataset_arg':
        #                 {'data_path': r'./resources/data/fit/rnnv3_nolabel/test%d_data.pkl',
        #                  'label_path': r'./resources/data/fit/rnnv3_nolabel/test%d_label.pkl',
        #                  # 'subseq_len': subseq_len,
        #                  # 'batch_size': batch_size,
        #                  # 'random_seed': random_seed,
        #                  # 'binary': False,
        #                  },
        #             'batch_size': batch_size,
        #             'shuffle': False,
        #             'num_workers': batch_size,
        #             'drop_last': False}
        data_arg = {'dataset': 'nn.dataset.subseq_feeder.SubseqFeeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnnv3_nolabel/train%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnnv3_nolabel/train%d_label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         },
                    'test_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnnv3_nolabel/test%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnnv3_nolabel/test%d_label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': batch_size,
                    'drop_last': False}
        loss_arg = {'loss_type': 'nn.loss.valid_itv_loss.ValidIntervalNLLLoss', 'weight': [1., 10., 1.]}
        pred_arg = {'pred_type': 'nn.pred.valid_itv_pred.ValidIntervalMultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 32}
        train_arg = {'epoch': 128, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'default',
                       'collate_fn': 'nn.dataset.collate_fn.collate_same_data_len_seq2seq',
                       'cal_acc_func': 'nn.metrics.valid_itv_metrics.cal_acc_func_valid_itv',
                       'cal_cm_func': 'nn.metrics.valid_itv_metrics.cal_cm_func_valid_itv',
                       'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn_arg': {'theta': 10},
                       'train_extra': {'teacher_forcing_ratio': 0.5},
                       'test_extra': {'teacher_forcing_ratio': 0},
                       }, f)
        train_with_config(config_path, format_config=True, folds=1)


if __name__ == '__main__':
    setting_name = 'cganv1_lr0.1'
    train_cganv1(setting_name)
    # setting_name = 'seq2seq_lr0.1'
    # train_seq2seq(setting_name)
    # setting_name = 'rnnv3_nolabel_lr0.1'
    # train_rnnv3_nolabel(setting_name)
    # setting_name = 'mlpv2_nolabel_lr0.1'
    # train_mlpv2_nolabel(setting_name)
