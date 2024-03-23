import yaml

from nn.train import train_with_config
from train_scripts import train_script_cddpm


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
        loss_arg = {'loss_type': 'CrossEntropyLoss'}
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
        loss_arg = {'loss_type': 'CrossEntropyLoss'}
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
        loss_arg = {'loss_type': 'CrossEntropyLoss', 'weight': [1., 10., 10., 10.]}
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
        loss_arg = {'loss_type': 'CrossEntropyLoss'}
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
        loss_arg = {'loss_type': 'CrossEntropyLoss'}
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
        loss_arg = {'loss_type': 'CrossEntropyLoss'}
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
        loss_arg = {'loss_type': 'CrossEntropyLoss'}
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
        loss_arg = {'loss_type': 'CrossEntropyLoss'}
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


def train_seq2seq(setting_name='seq2seq'):
    batch_size = 4
    # 12 beats
    subseq_len = 12 * 8
    random_seed = 404

    embed_size = 16
    hidden_size = 256
    num_class = 3
    for lr in [0.01]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': 'nn.net.seq2seq.Seq2Seq',
            'encoder_args': {
                'cond_data_feature_dim': 514,
                'hidden_size': hidden_size,
                'n_layers': 2,
                'dropout': 0.5,
            },
            'decoder_args': {
                'embed_size': embed_size,
                'hidden_size': hidden_size,
                'output_size': num_class + 2,
                'n_layers': 1,
                'dropout': 0.5,
            },
            'num_class': num_class,
            'embed_size': embed_size,
        }
        optimizer_arg = {'optimizer_type': 'Adam', 'lr': lr}
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
                        {'data_path': r'./resources/data/fit/label_pos/data.pkl',
                         'label_path': r'./resources/data/fit/label_pos/label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         'ho_pos': True,
                         'take_first': 100,
                        },
                    # 'test_dataset_arg':
                    #     {'data_path': r'./resources/data/fit/rnnv3_nolabel/data.pkl',
                    #      'label_path': r'./resources/data/fit/rnnv3_nolabel/label.pkl',
                    #      'subseq_len': subseq_len,
                    #      'random_seed': random_seed,
                    #      'use_random_iter': True,
                    #      'binary': False,
                    #      },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': batch_size,
                    'drop_last': False}
        loss_arg = {'loss_type': 'nn.loss.valid_itv_loss.ValidIntervalNLLLoss', 'weight': [1., 1., 1.]}
        pred_arg = {'pred_type': 'nn.pred.valid_itv_pred.ValidIntervalMultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 1}
        train_arg = {'epoch': 128, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'seq2seq',
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'cal_acc_func': 'nn.metrics.valid_itv_metrics.cal_acc_func_valid_itv',
                       'cal_cm_func': 'nn.metrics.valid_itv_metrics.cal_cm_func_valid_itv',
                       'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 10},
                       'train_extra': {'teacher_forcing_ratio': 0.5},
                       'test_extra': {'teacher_forcing_ratio': 0},
                       }, f)
        train_with_config(config_path, format_config=True, folds=1)


def train_cganv1(setting_name):
    random_seed = 404

    switch_label = False
    if switch_label:
        num_classes = 4
        weight = [1, 10, 10, 10]
    else:
        num_classes = 3
        weight = None
    epoch = 128
    scheduler_step_size = 64

    snap_feature = 514
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor

    batch_size = 6

    subseq_len = sample_beats * snap_divisor

    noise_feature_num = sample_snaps * num_classes
    compressed_channels = 16
    meta_feature_num = 2  # bpm, speed_stars

    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv1.Generator', 'nn.net.cganv1.Discriminator'],
            'params': [
                {
                    'output_len': sample_snaps,
                    'in_channels': snap_feature,
                    'num_classes': num_classes,
                    'noise_feature_num': noise_feature_num,
                    'compressed_channels': compressed_channels,
                },
                {
                    'output_len': sample_snaps,
                    'in_channels': snap_feature,
                    'num_classes': num_classes,
                    'compressed_channels': compressed_channels,
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
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 16}
        train_arg = {'epoch': epoch, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'gan',
                       'num_classes': num_classes,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv1_binary(setting_name):
    random_seed = 404

    switch_label = False
    binary = True

    if switch_label:
        num_classes = 4
        weight = [1, 10, 10, 10]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 128
    scheduler_step_size = 64

    snap_feature = 514
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor

    batch_size = 6

    subseq_len = sample_beats * snap_divisor

    noise_feature_num = sample_snaps * num_classes
    compressed_channels = 16
    meta_feature_num = 2  # bpm, speed_stars

    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv1.Generator', 'nn.net.cganv1.Discriminator'],
            'params': [
                {
                    'output_len': sample_snaps,
                    'in_channels': snap_feature,
                    'num_classes': num_classes,
                    'noise_feature_num': noise_feature_num,
                    'compressed_channels': compressed_channels,
                },
                {
                    'output_len': sample_snaps,
                    'in_channels': snap_feature,
                    'num_classes': num_classes,
                    'compressed_channels': compressed_channels,
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
        data_arg = {'dataset': 'nn.dataset.subseq_feeder.SubseqFeeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnnv3_nolabel_binary/train%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnnv3_nolabel_binary/train%d_label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         },
                    'test_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnnv3_nolabel_binary/test%d_data.pkl',
                         'label_path': r'./resources/data/fit/rnnv3_nolabel_binary/test%d_label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 16}
        train_arg = {'epoch': epoch, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'gan',
                       'num_classes': num_classes,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv2(setting_name):
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 10, 10, 10]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 128
    scheduler_step_size = 64

    snap_feature = 514
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor

    batch_size = 6

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv2.Generator', 'nn.net.cganv2.Discriminator'],
            'params': [
                {
                    'output_len': sample_snaps,
                    'in_channels': snap_feature,
                    'num_classes': num_classes,
                    'noise_in_channels': num_classes,
                    'compressed_channels': compressed_channels,
                },
                {
                    'output_len': sample_snaps,
                    'in_channels': snap_feature,
                    'num_classes': num_classes,
                    'compressed_channels': compressed_channels,
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
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 16}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': True}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'gan',
                       'num_classes': num_classes,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv3(setting_name):
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 128
    scheduler_step_size = 64

    snap_feature = 514
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor

    batch_size = 6

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv3.Generator', 'nn.net.cganv3.Discriminator'],
            'params': [
                {
                    'output_len': sample_snaps,
                    'in_channels': snap_feature,
                    'num_classes': num_classes,
                    'noise_in_channels': num_classes,
                    'compressed_channels': compressed_channels,
                },
                {
                    'output_len': sample_snaps,
                    'in_channels': snap_feature,
                    'num_classes': num_classes,
                    'compressed_channels': compressed_channels,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['SGD', 'SGD'],
            'params': [
                {'lr': lr,},
                {'lr': lr,},
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['StepLR', 'StepLR'],
            'params': [
                {'step_size': 128, 'gamma': 0.3},
                {'step_size': 128, 'gamma': 0.1},
            ]
        }
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
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 16}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'gan',
                       'num_classes': num_classes,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv3_pe(setting_name):
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 128
    scheduler_step_size = 64

    snap_feature = 514
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor

    batch_size = 6

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv3_pe.Generator', 'nn.net.cganv3_pe.Discriminator'],
            'params': [
                {
                    'output_len': sample_snaps,
                    'in_channels': snap_feature,
                    'num_classes': num_classes,
                    'noise_in_channels': num_classes,
                    'compressed_channels': compressed_channels,
                },
                {
                    'output_len': sample_snaps,
                    'in_channels': snap_feature,
                    'num_classes': num_classes,
                    'compressed_channels': compressed_channels,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['SGD', 'SGD'],
            'params': [
                {'lr': lr,},
                {'lr': lr,},
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['StepLR', 'StepLR'],
            'params': [
                {'step_size': 128, 'gamma': 0.3},
                {'step_size': 128, 'gamma': 0.1},
            ]
        }
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
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 16}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'gan',
                       'num_classes': num_classes,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv4(setting_name):
    """
    use slider and circle label
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 128
    scheduler_step_size = 64

    # mel features 4 * 128 + bpm 1 + speed_star 1
    snap_feature = 514
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor

    batch_size = 6

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv4.Generator', 'nn.net.cganv4.Discriminator'],
            'params': [
                {
                    'output_len': sample_snaps,
                    'in_channels': snap_feature,
                    'num_classes': num_classes,
                    'noise_in_channels': num_classes,
                    'compressed_channels': compressed_channels,
                },
                {
                    'output_len': sample_snaps,
                    'in_channels': snap_feature,
                    'num_classes': num_classes,
                    'compressed_channels': compressed_channels,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['SGD', 'SGD'],
            'params': [
                {'lr': lr,},
                {'lr': lr,},
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['StepLR', 'StepLR'],
            'params': [
                {'step_size': 16, 'gamma': 0.3},
                {'step_size': 128, 'gamma': 0.1},
            ]
        }
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
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'gan',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn_arg': {'theta': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_seqganv1(setting_name='seqganv1'):
    """
    use slider and circle label
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 256
    generator_pretrain_epoch = 64
    discriminator_pretrain_epoch = 64
    scheduler_step_size = 256

    snap_feature = 514
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor

    batch_size = 16

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    embedding_dim = 256
    hidden_dim = 512

    for lr in [0.01]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.seqganv1.Generator', 'nn.net.seqganv1.Discriminator'],
            'params': [
                {
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'cond_data_feature_dim': snap_feature,
                    'vocab_size': num_classes,
                    'num_layers': 2,
                },
                {
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'cond_data_feature_dim': snap_feature,
                    'vocab_size': num_classes,
                    'dropout': 0.2,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['Adam', 'Adam'],
            'params': [
                {'lr': lr,},
                {'lr': lr,},
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['StepLR', 'StepLR'],
            'params': [
                {'step_size': 16, 'gamma': 0.3},
                {'step_size': 128, 'gamma': 0.1},
            ]
        }
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
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'NLLLoss',
            'BCELoss',
        ],
            'params': [
                {},
                {},
            ]
        }
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'generator_pretrain_epoch': generator_pretrain_epoch,
                     'discriminator_pretrain_epoch': discriminator_pretrain_epoch,}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'rnngan_with_pretrain',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       # 'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn': None,
                       'grad_alter_fn_arg': {'theta': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_seqganv2(setting_name='seqganv2'):
    """
    use slider and circle label
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 64
    generator_pretrain_epoch = 1
    discriminator_pretrain_epoch = 1
    scheduler_step_size = 256

    adv_generator_epoch = 1
    adv_discriminator_epoch = 1

    snap_feature = 514
    snap_divisor = 8
    sample_beats = 12
    sample_snaps = sample_beats * snap_divisor

    batch_size = 16

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    embedding_dim = 128
    hidden_dim = 512

    for gen_lr_mle, gen_lr, dis_lr in [(0.01, 0.0001, 0.001)]:
        print('init gen_lr %s, dis_lr %s' % (str(gen_lr), str(dis_lr)))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.seqganv2.Generator', 'nn.net.seqganv2.Discriminator'],
            'params': [
                {
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'cond_data_feature_dim': snap_feature,
                    'vocab_size': num_classes,
                    'seq_len': snap_divisor,
                    'num_layers': 3,
                },
                {
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'cond_data_feature_dim': snap_feature,
                    'vocab_size': num_classes,
                    'seq_len': snap_divisor,
                    'dropout': 0.2,
                    'num_layers': 3,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['Adam', 'Adam', 'Adam'],
            'params': [
                {'lr': gen_lr_mle,},
                {'lr': gen_lr,},
                {'lr': dis_lr,},
            ],
            'models_index': [
                0, 0, 1
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['StepLR', 'StepLR', 'StepLR'],
            'params': [
                {'step_size': 16, 'gamma': 0.3},
                {'step_size': 16, 'gamma': 0.3},
                {'step_size': 128, 'gamma': 0.1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.seqganv2_feeder.SeqGANv2Feeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnnv3_nolabel/data.pkl',
                         'label_path': r'./resources/data/fit/rnnv3_nolabel/label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         'take_first': 250,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'NLLLoss',
            'BCELoss',
        ],
            'params': [
                {},
                {},
            ]
        }
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'generator_pretrain_epoch': generator_pretrain_epoch,
                     'discriminator_pretrain_epoch': discriminator_pretrain_epoch,
                     'adv_generator_epoch': adv_generator_epoch,
                     'adv_discriminator_epoch': adv_discriminator_epoch,
                     'adaptive_adv_train': True,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'rnngan_with_pretrain',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       # 'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn': None,
                       'grad_alter_fn_arg': {'theta': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_seqganv3(setting_name='seqganv3'):
    """
    use slider and circle label
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 64
    generator_pretrain_epoch = 0
    discriminator_pretrain_epoch = 0
    scheduler_step_size = 256

    snap_feature = 514
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor

    batch_size = 16

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    embedding_dim = 128
    hidden_dim = 512

    for lr in [0.01]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.seqganv3.Generator', 'nn.net.seqganv3.Discriminator'],
            'params': [
                {
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'cond_data_feature_dim': snap_feature,
                    'vocab_size': num_classes,
                    'seq_len': snap_divisor,
                    'num_layers': 2,
                },
                {
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'cond_data_feature_dim': snap_feature,
                    'vocab_size': num_classes,
                    'seq_len': snap_divisor,
                    'dropout': 0.2,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['Adam', 'Adam'],
            'params': [
                {'lr': lr,},
                {'lr': lr,},
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['StepLR', 'StepLR'],
            'params': [
                {'step_size': 16, 'gamma': 0.3},
                {'step_size': 128, 'gamma': 0.1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.subseq_feeder.SubseqFeeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/label_pos/train%d_data.pkl',
                         'label_path': r'./resources/data/fit/label_pos/train%d_label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         },
                    'test_dataset_arg':
                        {'data_path': r'./resources/data/fit/label_pos/test%d_data.pkl',
                         'label_path': r'./resources/data/fit/label_pos/test%d_label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'MSELoss',
            'BCELoss',
        ],
            'params': [
                {},
                {},
            ]
        }
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'generator_pretrain_epoch': generator_pretrain_epoch,
                     'discriminator_pretrain_epoch': discriminator_pretrain_epoch,}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'seqgan_adv_loss',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       # 'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn': None,
                       'grad_alter_fn_arg': {'theta': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_seqganv3_dis_deep(setting_name='seqganv3_dis_deep'):
    """
    use slider and circle label
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 64
    generator_pretrain_epoch = 4
    discriminator_pretrain_epoch = 1
    scheduler_step_size = 256

    adv_generator_epoch = 1
    adv_discriminator_epoch = 1

    snap_feature = 514
    snap_divisor = 8
    sample_beats = 12
    sample_snaps = sample_beats * snap_divisor

    batch_size = 16

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    embedding_dim = 128
    hidden_dim = 512

    for gen_lr_mle, gen_lr, dis_lr in [(0.01, 0.01, 0.01)]:
        print('init gen_lr %s, dis_lr %s' % (str(gen_lr), str(dis_lr)))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.seqganv3.Generator', 'nn.net.seqganv3.Discriminator'],
            'params': [
                {
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'cond_data_feature_dim': snap_feature,
                    'vocab_size': num_classes,
                    'seq_len': snap_divisor,
                    'num_layers': 2,
                },
                {
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'cond_data_feature_dim': snap_feature,
                    'vocab_size': num_classes,
                    'seq_len': snap_divisor,
                    'dropout': 0.2,
                    'num_layers': 3,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['Adam', 'Adam', 'Adam'],
            'params': [
                {'lr': gen_lr_mle,},
                {'lr': gen_lr,},
                {'lr': dis_lr,},
            ],
            'models_index': [
                0, 0, 1
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['StepLR', 'StepLR', 'StepLR'],
            'params': [
                {'step_size': 16, 'gamma': 0.3},
                {'step_size': 16, 'gamma': 0.3},
                {'step_size': 128, 'gamma': 0.1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.subseq_feeder.SubseqFeeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/label_pos/train%d_data.pkl',
                         'label_path': r'./resources/data/fit/label_pos/train%d_label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         },
                    'test_dataset_arg':
                        {'data_path': r'./resources/data/fit/label_pos/test%d_data.pkl',
                         'label_path': r'./resources/data/fit/label_pos/test%d_label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'MSELoss',
            'BCELoss',
        ],
            'params': [
                {},
                {},
            ]
        }
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(gen_lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(gen_lr) + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'generator_pretrain_epoch': generator_pretrain_epoch,
                     'discriminator_pretrain_epoch': discriminator_pretrain_epoch,
                     'adv_generator_epoch': adv_generator_epoch,
                     'adv_discriminator_epoch': adv_discriminator_epoch,
                     'adv_loss_multiplier': 1.,
                     'adaptive_adv_train': True,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'seqgan_adv_loss',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       # 'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn': 'value',
                       'grad_alter_fn_arg': {'clip_value': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_vae(setting_name='vae'):
    """
    use slider and circle label
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 64
    generator_pretrain_epoch = 4
    discriminator_pretrain_epoch = 1
    scheduler_step_size = 256

    adv_generator_epoch = 1
    adv_discriminator_epoch = 1

    snap_feature = 514
    snap_divisor = 8
    sample_beats = 12
    sample_snaps = sample_beats * snap_divisor

    batch_size = 16

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    embedding_dim = 128
    hidden_dim = 512

    for lr in [0.001]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': 'nn.net.vae.VAE',
            'hidden_dim': hidden_dim,
            'cond_data_feature_dim': snap_feature,
            'num_class': num_classes,
            'seq_len': sample_snaps,
            'num_layers': 2,
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': 'Adam',
            'lr': lr,
        }
        scheduler_arg = {
            'scheduler_type': 'StepLR',
            'step_size': 16, 'gamma': 0.3,
        }
        data_arg = {'dataset': 'nn.dataset.subseq_feeder.SubseqFeeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnnv3_nolabel/data.pkl',
                         'label_path': r'./resources/data/fit/rnnv3_nolabel/label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         'take_first': 100,
                         },
                    'test_dataset_arg':
                        {'data_path': r'./resources/data/fit/rnnv3_nolabel/data.pkl',
                         'label_path': r'./resources/data/fit/rnnv3_nolabel/label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'MSELoss',
            'BCELoss',
        ],
            'params': [
                {},
                {},
            ]
        }
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'vae',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       # 'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'clip_value': 10},
                       'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_vaev2(setting_name='vaev2'):
    """
    use slider and circle label
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 64
    generator_pretrain_epoch = 4
    discriminator_pretrain_epoch = 1
    scheduler_step_size = 256

    adv_generator_epoch = 1
    adv_discriminator_epoch = 1

    snap_feature = 514
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor

    batch_size = 16

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    embedding_dim = 128
    hidden_dim = 512

    for lr in [0.0001]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': 'nn.net.vaev2.VAE',
            'hidden_dim': hidden_dim,
            'cond_data_feature_dim': snap_feature,
            'num_class': num_classes,
            'seq_len': sample_snaps,
            'num_layers': 2,
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': 'Adam',
            'lr': lr,
        }
        scheduler_arg = {
            'scheduler_type': 'StepLR',
            'step_size': 16, 'gamma': 0.3,
        }
        data_arg = {'dataset': 'nn.dataset.subseq_feeder.SubseqFeeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/label_pos/data.pkl',
                         'label_path': r'./resources/data/fit/label_pos/label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         'take_first': 500,
                         'ho_pos': True,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'MSELoss',
            'BCELoss',
        ],
            'params': [
                {},
                {},
            ]
        }
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'vae',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       # 'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'clip_value': 10},
                       'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_vaev3(setting_name='vaev3_new'):
    """
    use density instead of discrete labels
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 64
    generator_pretrain_epoch = 4
    discriminator_pretrain_epoch = 1
    scheduler_step_size = 256

    adv_generator_epoch = 1
    adv_discriminator_epoch = 1

    snap_feature = 514
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor

    batch_size = 16

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    embedding_dim = 128
    hidden_dim = 512

    for lr in [0.001]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': 'nn.net.vaev3.VAE',
            'hidden_dim': hidden_dim,
            'cond_data_feature_dim': snap_feature,
            'in_channels': 4,
            'pos_emb_period': snap_divisor,
            'pos_emb_channels': 4,
            'enc_layers': 2,
            'dec_layers': 1,
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': 'Adam',
            'lr': lr,
        }
        scheduler_arg = {
            'scheduler_type': 'StepLR',
            'step_size': 16, 'gamma': 0.3,
        }
        data_arg = {'dataset': 'nn.dataset.subseq_heatmap_feeder.SubseqFeeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/label_pos/data.pkl',
                         'label_path': r'./resources/data/fit/label_pos/label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         'take_first': 500,
                         'ho_pos': True,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'MSELoss',
            'BCELoss',
        ],
            'params': [
                {},
                {},
            ]
        }
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'init_teacher_forcing_ratio': 0.5,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'vae',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       # 'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'clip_value': 10},
                       'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_vaev4(setting_name='vaev4'):
    """
    a beat for each rnn step
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 64
    generator_pretrain_epoch = 4
    discriminator_pretrain_epoch = 1
    scheduler_step_size = 256

    adv_generator_epoch = 1
    adv_discriminator_epoch = 1

    snap_feature = 514
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor

    batch_size = 16

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    embedding_dim = 128
    hidden_dim = 512

    for lr in [0.001]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': 'nn.net.vaev4.VAE',
            'hidden_dim': hidden_dim,
            'cond_data_feature_dim': snap_feature,
            'in_channels': 4,
            'step_snaps': snap_divisor,
            'pos_emb_period': sample_beats,
            'pos_emb_channels': 8,
            'enc_layers': 2,
            'dec_layers': 2,
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': 'Adam',
            'lr': lr,
        }
        scheduler_arg = {
            'scheduler_type': 'StepLR',
            'step_size': 16, 'gamma': 0.3,
        }
        data_arg = {'dataset': 'nn.dataset.subseq_heatmap_feeder.SubseqFeeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/label_pos/data.pkl',
                         'label_path': r'./resources/data/fit/label_pos/label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         'take_first': 500,
                         'ho_pos': True,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'MSELoss',
            'BCELoss',
        ],
            'params': [
                {},
                {},
            ]
        }
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'init_teacher_forcing_ratio': 0.5,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'vae',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       # 'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'clip_value': 10},
                       'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv5(setting_name='cganv5'):
    """
    use slider and circle label
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 128
    scheduler_step_size = 64

    # mel features 4 * 128 + bpm 1 + speed_star 1
    snap_feature = 514
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor

    batch_size = 6

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.1, 0.1]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv5.Generator', 'nn.net.cganv5.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'label_dim': 4,
                    'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                },
                {
                    'seq_len': subseq_len,
                    'label_dim': 4,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                },
            ]
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': ['SGD', 'SGD'],
            'params': [
                {'lr': gen_lr,},
                {'lr': dis_lr,},
            ]
        }
        scheduler_arg = {
            'scheduler_type': ['StepLR', 'StepLR'],
            'params': [
                {'step_size': 16, 'gamma': 0.3},
                {'step_size': 128, 'gamma': 0.1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.subseq_heatmap_feeder.SubseqFeeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/label_pos/data.pkl',
                         'label_path': r'./resources/data/fit/label_pos/label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         'take_first': None,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 1,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 3,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'wgan',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': 'util.net_util.grad_clipping',
                       'grad_alter_fn_arg': {'theta': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv6(setting_name='cganv6'):
    """
    use slider and circle label
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 2048
    scheduler_step_size = 64

    # mel features 4 * 128 + bpm 1 + speed_star 1
    snap_feature = 517
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor

    batch_size = 8

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.001, 0.001]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv6.Generator', 'nn.net.cganv6.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'label_dim': 4,
                    'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                },
                {
                    'seq_len': subseq_len,
                    'label_dim': 4,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
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
            'scheduler_type': ['StepLR', 'StepLR'],
            'params': [
                {'step_size': 16, 'gamma': 0.3},
                {'step_size': 128, 'gamma': 0.1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.subseq_heatmap_feeder.SubseqFeeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/label_pos_v2/data.pkl',
                         'label_path': r'./resources/data/fit/label_pos_v2/label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         'shuffle': True,
                         'take_first': None,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 32}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 3,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 5,
                     'lambda_gp': 10,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'wgan',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 1},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv6_within_batch(setting_name='cganv6_within_batch'):
    """
    use slider and circle label
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 2048
    scheduler_step_size = 64

    # mel features 4 * 128 + bpm 1 + speed_star 1
    snap_feature = 517
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor

    batch_size = 8

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.001, 0.001]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv6.Generator', 'nn.net.cganv6.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'label_dim': 4,
                    'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                },
                {
                    'seq_len': subseq_len,
                    'label_dim': 4,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
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
            'scheduler_type': ['StepLR', 'StepLR'],
            'params': [
                {'step_size': 16, 'gamma': 0.3},
                {'step_size': 128, 'gamma': 0.1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.subseq_heatmap_feeder.SubseqFeeder',
                    'train_dataset_arg':
                        {'data_path': r'./resources/data/fit/label_pos_v2/data.pkl',
                         'label_path': r'./resources/data/fit/label_pos_v2/label.pkl',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'binary': False,
                         'shuffle': True,
                         'take_first': None,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 32}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 3,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 5,
                     'adv_discriminator_epoch': 1,
                     'lambda_gp': 10,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'wganwb',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 1},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv7_within_batch_whole_seq(setting_name='cganv7_within_batch_whole_seq'):
    """
    use slider and circle label
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 2048
    scheduler_step_size = 64

    # mel features 4 * 128 + bpm 1 + speed_star 1
    # snap_feature = 517
    snap_feature = 133
    snap_divisor = 8
    sample_beats = 48
    sample_snaps = sample_beats * snap_divisor
    snap_data_len = 4

    batch_size = 16

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.1, 0.1]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv7.Generator', 'nn.net.cganv7.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len * snap_data_len,
                    'label_dim': 4,
                    'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'normalize': 'LN',
                },
                {
                    'seq_len': subseq_len * snap_data_len,
                    'label_dim': 4,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'normalize': 'LN',
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
            'scheduler_type': ['StepLR', 'StepLR'],
            'params': [
                {'step_size': 16, 'gamma': 0.3},
                {'step_size': 128, 'gamma': 0.1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.whole_seq_feeder.WholeSeqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'./resources/data/fit/label_pos_v3',
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'data_process_dim': 128,
                         'binary': False,
                         'take_first': None,
                         },
                    'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 4,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 32}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 10,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 0.1,
                     'adv_discriminator_epoch': 1.,
                     'lambda_gp': 10,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'wganwb',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)
# def train_cganv7_within_batch(setting_name='cganv7_within_batch'):
#     """
#     use slider and circle label
#     """
#     random_seed = 404
#
#     switch_label = False
#     binary = False
#
#     if switch_label:
#         num_classes = 4
#         weight = [1, 1, 1, 1]
#     else:
#         if binary:
#             num_classes = 2
#             weight = None
#         else:
#             num_classes = 3
#             weight = None
#     epoch = 2048
#     scheduler_step_size = 64
#
#     # mel features 4 * 128 + bpm 1 + speed_star 1
#     # snap_feature = 517
#     snap_feature = 133
#     snap_divisor = 8
#     sample_beats = 48
#     sample_snaps = sample_beats * snap_divisor
#     snap_data_len = 4
#
#     batch_size = 8
#
#     subseq_len = sample_beats * snap_divisor
#
#     compressed_channels = 16
#
#     for gen_lr, dis_lr in [[0.001, 0.001]]:
#         print('init lr %s' % str(gen_lr))
#         config_path = './resources/config/train/%s.yaml' % setting_name
#         model_arg = {
#             'model_type': ['nn.net.cganv7.Generator', 'nn.net.cganv7.Discriminator'],
#             'params': [
#                 {
#                     'seq_len': subseq_len * snap_data_len,
#                     'label_dim': 4,
#                     'noise_dim': 64,
#                     'cond_data_feature_dim': snap_feature,
#                 },
#                 {
#                     'seq_len': subseq_len * snap_data_len,
#                     'label_dim': 4,
#                     # 'noise_dim': 64,
#                     'cond_data_feature_dim': snap_feature,
#                 },
#             ]
#         }  # , 'num_block': [1, 1, 1, 1]
#         optimizer_arg = {
#             'optimizer_type': ['RMSprop', 'RMSprop'],
#             'params': [
#                 {'lr': gen_lr,},
#                 {'lr': dis_lr,},
#             ]
#         }
#         scheduler_arg = {
#             'scheduler_type': ['StepLR', 'StepLR'],
#             'params': [
#                 {'step_size': 16, 'gamma': 0.3},
#                 {'step_size': 128, 'gamma': 0.1},
#             ]
#         }
#         data_arg = {'dataset': 'nn.dataset.subseq_heatmap_feederv2.SubseqFeeder',
#                     'train_dataset_arg':
#                         {'data_path': r'./resources/data/fit/label_pos_v3/data.pkl',
#                          'label_path': r'./resources/data/fit/label_pos_v3/label.pkl',
#                          'subseq_len': subseq_len,
#                          'random_seed': random_seed,
#                          'use_random_iter': True,
#                          'snap_data_len': 4,
#                          'binary': False,
#                          'shuffle': True,
#                          'take_first': None,
#                          },
#                     'batch_size': batch_size,
#                     'shuffle': False,
#                     'num_workers': 1,
#                     'drop_last': False}
#         loss_arg = {'loss_type': [
#             # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
#             'CrossEntropyLoss',
#             'CrossEntropyLoss',
#         ],
#             'params': [
#                 {'weight': weight},
#                 {'weight': weight},
#             ]}
#         pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
#         output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
#                       'model_save_dir': './resources/result/' + setting_name + '/%d',
#                       'model_save_step': 32}
#         train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
#                      'discriminator_pretrain_epoch': 10,
#                      'adaptive_adv_train': False,
#                      'adv_generator_epoch': 10,
#                      'adv_discriminator_epoch': 1,
#                      'lambda_gp': 10,
#                      }
#         with open(config_path, 'w') as f:
#             yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
#                        'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
#                        'train_arg': train_arg,
#                        'output_device': 0,
#                        'train_type': 'wganwb',
#                        'num_classes': num_classes,
#                        'random_seed': random_seed,
#                        'collate_fn': 'nn.dataset.collate_fn.default_collate',
#                        'grad_alter_fn': None,
#                        # 'grad_alter_fn': 'norm',
#                        # 'grad_alter_fn_arg': {'max_norm': 10},
#                        'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
#                        'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
#                        }, f)
#         train_with_config(config_path, folds=1, format_config=True)
def train_cganv7_within_batch(setting_name='cganv7_within_batch'):
    """
    use slider and circle label
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 2048
    scheduler_step_size = 64

    # mel features 4 * 128 + bpm 1 + speed_star 1
    # snap_feature = 517
    snap_feature = 133
    snap_divisor = 8
    sample_beats = 144
    sample_snaps = sample_beats * snap_divisor
    snap_data_len = 4

    batch_size = 8

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.01, 0.01]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv7.Generator', 'nn.net.cganv7.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len * snap_data_len,
                    'label_dim': 4,
                    'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                },
                {
                    'seq_len': subseq_len * snap_data_len,
                    'label_dim': 4,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
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
                {'milestones': [25, 50, 350, 800], 'gamma': 0.3},
                {'milestones': [25, 50, 350, 800], 'gamma': 0.3},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.subseq_heatmap_feeder_separate.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'./resources/data/fit/label_pos_v3',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'snap_data_len': 4,
                         'data_process_dim': 128,
                         'binary': False,
                         'take_first': None,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 4,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 10,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'gen_lambda_step': [50, 100],
                     'gen_lambda': [0.05, 0.075, 0.1],
                     'lambda_gp': 10,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'train_type': 'wganwb',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv8_within_batch(setting_name='cganv8_within_batch'):
    """
    use slider and circle label
    """
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 20480
    scheduler_step_size = 64

    # mel features 4 * 128 + bpm 1 + speed_star 1
    # snap_feature = 517
    snap_feature = 133
    snap_divisor = 8
    sample_beats = 144
    sample_snaps = sample_beats * snap_divisor
    snap_data_len = 4

    batch_size = 8

    subseq_len = sample_beats * snap_divisor

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.0003, 0.001]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv8.Generator', 'nn.net.cganv8.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len * snap_data_len,
                    'label_dim': 4,
                    'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                },
                {
                    'seq_len': subseq_len * snap_data_len,
                    'label_dim': 4,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
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
                {'milestones': [120], 'gamma': 1},
                {'milestones': [120], 'gamma': 1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.subseq_heatmap_feeder_separate.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'./resources/data/fit/label_pos_v3',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'snap_data_len': 4,
                         'data_process_dim': 128,
                         'binary': False,
                         'take_first': None,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 2,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 8}
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 20,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0.03,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [200],
                     'gen_lambda': [0.1, 0.1],
                     'lambda_gp': 10,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'wganwb',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv9_within_batch(setting_name='cganv9_within_batch'):
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 20480
    scheduler_step_size = 64

    # mel features 4 * 128 + bpm 1 + speed_star 1
    # snap_feature = 517
    snap_feature = 69
    snap_divisor = 8
    sample_beats = 72
    sample_snaps = sample_beats * snap_divisor
    snap_data_len = 16

    batch_size = 8

    subseq_len = snap_data_len * sample_snaps  # 72 * 8 * 16

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.0003, 0.001]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv9.Generator', 'nn.net.cganv9.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'normalize': 'BN',
                    'middle_dim': 64,
                },
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'normalize': 'LN',
                    'middle_dim': 64,
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
                {'milestones': [25, 100, 150, 200, 250], 'gamma': 0.3},
                {'milestones': [25, 100, 150, 200, 250], 'gamma': 0.3},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.subseq_heatmap_feeder_separatev3.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'./resources/data/fit/label_pos_v5',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'data_process_dim': 48,  # do some preprocessing to data
                         'coeff_data_len': 1,
                         'coeff_label_len': 1,
                         'take_first': None,
                         'pad': False,
                         'level_coeff': 0,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 2,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 8,
                      'train_state_save_step': 8,
                      }
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 10,
                     'generator_pretrain_epoch': 20,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0.02,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [200],
                     'gen_lambda': [0.2, 0.2],
                     'noise_level_step': [100, 150, 200, 250],
                     'noise_level': [1, 0.7, 0.4, 0.2, 0.1],
                     'lambda_gp': 10,
                     'last_epoch': -1,
                     'last_phase': 'pretrain_generator',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'wganwb',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv10_within_batch(setting_name='cganv10_within_batch'):
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 20480
    scheduler_step_size = 64

    # mel features 4 * 128 + bpm 1 + speed_star 1
    # snap_feature = 517
    snap_feature = 69
    snap_divisor = 8
    sample_beats = 72
    sample_snaps = sample_beats * snap_divisor
    snap_data_len = 16

    batch_size = 8

    subseq_len = snap_data_len * sample_snaps  # 72 * 8 * 16

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.001, 0.01]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv10.Generator', 'nn.net.cganv10.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 128,
                },
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 64,
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
                {'milestones': [100, 200, 300], 'gamma': 0.3},
                {'milestones': [100, 200, 300], 'gamma': 0.3},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.subseq_heatmap_feeder_separatev3.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'./resources/data/fit/label_pos_v5',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'data_process_dim': 64,  # do some preprocessing to data
                         'coeff_data_len': 1,
                         'coeff_label_len': 1,
                         'take_first': None,
                         'pad': False,
                         'level_coeff': 0,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 2,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 8,
                      'train_state_save_step': 8,
                      }
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 15,
                     'generator_pretrain_epoch': 1,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0.02,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [200],
                     'gen_lambda': [0.2, 0.2],
                     'noise_level_step': [(100 + 25 * i) for i in range(9)],
                     'noise_level': [(1 - 0.1 * i) for i in range(10)],
                     'lambda_gp': 10,
                     'last_epoch': 95,
                     'last_phase': 'adversarial',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'wganwb',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv11_within_batch(setting_name='cganv11_within_batch'):
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 20480
    scheduler_step_size = 64

    # mel features 4 * 128 + bpm 1 + speed_star 1
    # snap_feature = 517
    snap_feature = 69
    snap_divisor = 8
    sample_beats = 72
    sample_snaps = sample_beats * snap_divisor
    snap_data_len = 16

    batch_size = 8

    subseq_len = snap_data_len * sample_snaps  # 72 * 8 * 16

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.001, 0.01]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv11.Generator', 'nn.net.cganv11.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 128,
                },
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 64,
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
                {'milestones': [50, 100, 150], 'gamma': 0.3},
                {'milestones': [100, 200, 300], 'gamma': 1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.subseq_heatmap_feeder_separatev3.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'./resources/data/fit/label_pos_v5',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'data_process_dim': 64,  # do some preprocessing to data
                         'coeff_data_len': 1,
                         'coeff_label_len': 1,
                         'take_first': None,
                         'pad': False,
                         'level_coeff': 0,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 2,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 8,
                      'train_state_save_step': 8,
                      }
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 25,
                     'generator_pretrain_epoch': 10,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0.02,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [200],
                     'gen_lambda': [0.2, 0.2],
                     'noise_level_step': [(50 + 25 * i) for i in range(9)],
                     'noise_level': [(1 - 0.1 * i) for i in range(10)],
                     'lambda_gp': 10,
                     'last_epoch': -1,
                     'last_phase': 'pretrain_generator',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'wganwb',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv12_within_batch(setting_name='cganv12_within_batch'):
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 20480
    scheduler_step_size = 64

    # mel features 4 * 128 + bpm 1 + speed_star 1
    # snap_feature = 517
    snap_feature = 69
    snap_divisor = 8
    sample_beats = 72
    sample_snaps = sample_beats * snap_divisor
    snap_data_len = 16

    batch_size = 8

    subseq_len = snap_data_len * sample_snaps  # 72 * 8 * 16

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.0001, 0.0001]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv12.Generator', 'nn.net.cganv12.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 128,
                },
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 64,
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
                {'milestones': [50, 100, 150], 'gamma': 0.3},
                {'milestones': [50, 100, 150], 'gamma': 0.3},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.subseq_heatmap_feeder_separatev3.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'./resources/data/fit/label_pos_v5',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'data_process_dim': 64,  # do some preprocessing to data
                         'coeff_data_len': 1,
                         'coeff_label_len': 1,
                         'take_first': None,
                         'pad': False,
                         'level_coeff': 0,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 2,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 8,
                      'train_state_save_step': 8,
                      }
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 50,
                     'generator_pretrain_epoch': 10,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0.02,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [200],
                     'gen_lambda': [0.1, 0.1],
                     # 'noise_level_step': [(50 + 25 * i) for i in range(9)],
                     # 'noise_level': [(1 - 0.1 * i) for i in range(10)],
                     'noise_level_step': [50],
                     'noise_level': [0, 0],
                     'lambda_gp': 10,
                     # 'last_epoch': 23,
                     # 'last_phase': 'adversarial',
                     # 'last_epoch': -1,
                     # 'last_phase': 'pretrain_generator',
                     'last_epoch': 9,
                     'last_phase': 'pretrain_generator',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'wganwb',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv13_within_batch(setting_name='cganv13_within_batch'):
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 20480
    scheduler_step_size = 64

    # mel features 4 * 128 + bpm 1 + speed_star 1
    # snap_feature = 517
    snap_feature = 69
    snap_divisor = 8
    sample_beats = 72
    sample_snaps = sample_beats * snap_divisor
    snap_data_len = 16

    batch_size = 8

    subseq_len = snap_data_len * sample_snaps  # 72 * 8 * 16

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.00001, 0.00003]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv13.Generator', 'nn.net.cganv13.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    'noise_dim': 16,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'preprocess_dim': 16,
                },
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'preprocess_dim': 16,
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
                {'milestones': [50], 'gamma': 0.3},
                {'milestones': [50], 'gamma': 0.3},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.subseq_heatmap_feeder_separatev3.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'./resources/data/fit/label_pos_v5',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'data_process_dim': 64,  # do some preprocessing to data
                         'coeff_data_len': 1,
                         'coeff_label_len': 1,
                         'take_first': None,
                         'pad': False,
                         'level_coeff': 0,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 2,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 8,
                      'train_state_save_step': 8,
                      }
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 25,
                     'generator_pretrain_epoch': 1,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0.02,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [1000],
                     'gen_lambda': [0.2, 0.2],
                     # 'noise_level_step': [(50 + 25 * i) for i in range(9)],
                     # 'noise_level': [(1 - 0.1 * i) for i in range(10)],
                     'noise_level_step': [50, 100],
                     'noise_level': [1, 1, 1],
                     'lambda_gp': 10,
                     # 'last_epoch': 199,
                     # 'last_phase': 'adversarial',
                     # 'last_epoch': 1,
                     # 'last_phase': 'pretrain_generator',
                     'last_epoch': -1,
                     'last_phase': 'pretrain_generator',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'wganwb',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv14_within_batch(setting_name='cganv13_within_batch'):
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 20480
    scheduler_step_size = 64

    # mel features 4 * 128 + bpm 1 + speed_star 1
    # snap_feature = 517
    snap_feature = 69
    snap_divisor = 8
    sample_beats = 72
    sample_snaps = sample_beats * snap_divisor
    snap_data_len = 16

    batch_size = 8

    subseq_len = snap_data_len * sample_snaps  # 72 * 8 * 16

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.0001, 0.0001]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv14.Generator', 'nn.net.cganv14.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    'noise_dim': 16,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'preprocess_dim': 16,
                },
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'preprocess_dim': 16,
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
                {'milestones': [50], 'gamma': 1},
                {'milestones': [50], 'gamma': 1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.subseq_heatmap_feeder_separatev3.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'./resources/data/fit/label_pos_v5',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'data_process_dim': 64,  # do some preprocessing to data
                         'coeff_data_len': 1,
                         'coeff_label_len': 1,
                         'take_first': None,
                         'pad': False,
                         'level_coeff': 0,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 2,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 8,
                      'train_state_save_step': 8,
                      }
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 10,
                     'generator_pretrain_epoch': 1,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0.02,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [1000],
                     'gen_lambda': [1, 1],
                     # 'noise_level_step': [(50 + 25 * i) for i in range(9)],
                     # 'noise_level': [(1 - 0.1 * i) for i in range(10)],
                     'noise_level_step': [50, 100],
                     'noise_level': [1, 1, 1],
                     'lambda_gp': 10,
                     # 'last_epoch': 63,
                     # 'last_phase': 'adversarial',
                     # 'last_epoch': 1,
                     # 'last_phase': 'pretrain_generator',
                     'last_epoch': -1,
                     'last_phase': 'pretrain_generator',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'wganwb',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv14_within_batch_heatmapv1(setting_name='cganv14_heatmapv1'):
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 20480
    scheduler_step_size = 64

    num_meta = 11
    snap_feature = 64 + num_meta
    snap_divisor = 8
    sample_beats = 72
    sample_snaps = sample_beats * snap_divisor
    # 166 frames/s
    snap_data_len = 16

    batch_size = 8

    # ~56s per subseq
    subseq_len = snap_data_len * sample_snaps  # 72 * 8 * 16

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.00003, 0.00003]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv14.Generator', 'nn.net.cganv14.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    'noise_dim': 16,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'preprocess_dim': 16,
                },
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'preprocess_dim': 16,
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
                {'milestones': [30], 'gamma': 1},
                {'milestones': [30], 'gamma': 1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.feeder_heatmapv1.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'./resources/data/fit/heatmapv1',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': None,
                         'pad': False,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 2,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 8,
                      'train_state_save_step': 1,
                      }
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 1,
                     'generator_pretrain_epoch': 1,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0.02,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [1000],
                     'gen_lambda': [1, 1],
                     # 'noise_level_step': [(50 + 25 * i) for i in range(9)],
                     # 'noise_level': [(1 - 0.1 * i) for i in range(10)],
                     'noise_level_step': [50, 100],
                     'noise_level': [1, 1, 1],
                     'lambda_gp': 10,
                     'last_epoch': 30,
                     'last_phase': 'adversarial',
                     # 'last_epoch': 3,
                     # 'last_phase': 'pretrain_discriminator',
                     # 'last_epoch': -1,
                     # 'last_phase': 'pretrain_generator',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'wganwb',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv15_within_batch_heatmapv1(setting_name='cganv15_heatmapv1'):
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 20480
    scheduler_step_size = 64

    num_meta = 11
    snap_feature = 64 + num_meta
    snap_divisor = 8
    sample_beats = 72
    sample_snaps = sample_beats * snap_divisor
    # 166 frames/s
    snap_data_len = 16

    batch_size = 8

    # ~56s per subseq
    subseq_len = snap_data_len * sample_snaps  # 72 * 8 * 16

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.00003, 0.00003]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv15.Generator', 'nn.net.cganv15.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    'noise_dim': 32,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'preprocess_dim': 32,
                },
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'preprocess_dim': 32,
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
                {'milestones': [30], 'gamma': 1},
                {'milestones': [30], 'gamma': 1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.feeder_heatmapv1.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'./resources/data/fit/heatmapv1',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': 5120,
                         'pad': False,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 2,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 8,
                      'train_state_save_step': 1,
                      }
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 1,
                     'generator_pretrain_epoch': 1,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 0.2,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0.02,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [1000],
                     'gen_lambda': [1, 1],
                     # 'noise_level_step': [(50 + 25 * i) for i in range(9)],
                     # 'noise_level': [(1 - 0.1 * i) for i in range(10)],
                     'noise_level_step': [50, 100],
                     'noise_level': [1, 1, 1],
                     'lambda_gp': 10,
                     # 'last_epoch': 51,
                     # 'last_phase': 'adversarial',
                     # 'last_epoch': 3,
                     # 'last_phase': 'pretrain_discriminator',
                     'last_epoch': -1,
                     'last_phase': 'pretrain_generator',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'wganwb',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_cganv16_within_batch_heatmapv1(setting_name='cganv15_heatmapv1'):
    random_seed = 404

    switch_label = False
    binary = False

    if switch_label:
        num_classes = 4
        weight = [1, 1, 1, 1]
    else:
        if binary:
            num_classes = 2
            weight = None
        else:
            num_classes = 3
            weight = None
    epoch = 20480
    scheduler_step_size = 64

    num_meta = 11
    snap_feature = 64 + num_meta
    snap_divisor = 8
    sample_beats = 72
    sample_snaps = sample_beats * snap_divisor
    # 166 frames/s
    snap_data_len = 16

    batch_size = 8

    # ~56s per subseq
    subseq_len = snap_data_len * sample_snaps  # 72 * 8 * 16

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.00001, 0.00001]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.cganv16.Generator', 'nn.net.cganv16.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    'noise_dim': 32,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'preprocess_dim': 32,
                },
                {
                    'seq_len': subseq_len,
                    'label_dim': 5,
                    # 'noise_dim': 64,
                    'cond_data_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 256,
                    'preprocess_dim': 32,
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
                {'milestones': [30], 'gamma': 1},
                {'milestones': [30], 'gamma': 1},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.feeder_heatmapv1.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'./resources/data/fit/heatmapv1',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': None,
                         'pad': False,
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 2,
                    'drop_last': False}
        loss_arg = {'loss_type': [
            # 'nn.loss.multi_pred_loss.MultiPredNLLLoss',
            'CrossEntropyLoss',
            'CrossEntropyLoss',
        ],
            'params': [
                {'weight': weight},
                {'weight': weight},
            ]}
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/%d',
                      'model_save_step': 8,
                      'train_state_save_step': 1,
                      }
        train_arg = {'epoch': epoch, 'eval_step': 1, 'use_ext_cond_data': False,
                     'discriminator_pretrain_epoch': 1,
                     'generator_pretrain_epoch': 1,
                     'adaptive_adv_train': False,
                     'adv_generator_epoch': 1,
                     'adv_discriminator_epoch': 1,
                     'log_exp_replay_prob': 0.02,
                     'exp_replay_wait': 5,
                     'gen_lambda_step': [1000],
                     'gen_lambda': [1, 1],
                     # 'noise_level_step': [(50 + 25 * i) for i in range(9)],
                     # 'noise_level': [(1 - 0.1 * i) for i in range(10)],
                     'noise_level_step': [50, 100],
                     'noise_level': [1, 1, 1],
                     'lambda_gp': 10,
                     # 'last_epoch': 12,
                     # 'last_phase': 'adversarial',
                     # 'last_epoch': 3,
                     # 'last_phase': 'pretrain_discriminator',
                     'last_epoch': -1,
                     'last_phase': 'pretrain_generator',
                     'num_models': 2,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'wganwb',
                       'num_classes': num_classes,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.default_collate',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=1, format_config=True)


def train_simple_acganv1_heatmapv2(setting_name='simple_acganv1_heatmapv2'):
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
    snap_feature = 40

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.00003, 0.00003]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.simple_acganv1.Generator', 'nn.net.simple_acganv1.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'tgt_dim': 5,
                    'noise_dim': 16,
                    'audio_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 128,
                    'preprocess_dim': 16,
                    'cls_label_dim': 3,
                },
                {
                    'seq_len': subseq_len,
                    'tgt_dim': 5,
                    # 'noise_dim': 64,
                    'audio_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 128,
                    'preprocess_dim': 16,
                    'cls_label_dim': 3,
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
                {'milestones': [50], 'gamma': 0.3},
                {'milestones': [50], 'gamma': 0.3},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.feeder_heatmapv2_meta.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'/home/data1/xiezheng/osu_mapper/preprocessed',
                         'subseq_len': subseq_len,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': 40960,
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
                     'noise_level_step': [25, 75],
                     'noise_level': [1, 0.3, 0.1],
                     'lambda_gp': 10,
                     'lambda_cls': 1.,
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


def train_simple_acganv2_heatmapv2(setting_name='simple_acganv2_heatmapv2'):
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
    snap_feature = 40

    compressed_channels = 16

    for gen_lr, dis_lr in [[0.0001, 0.0001]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.simple_acganv2.Generator', 'nn.net.simple_acganv2.Discriminator'],
            'params': [
                {
                    'seq_len': subseq_len,
                    'tgt_dim': 5,
                    'noise_dim': 16,
                    'audio_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 128,
                    'preprocess_dim': 16,
                    'cls_label_dim': 3,
                },
                {
                    'seq_len': subseq_len,
                    'tgt_dim': 5,
                    # 'noise_dim': 64,
                    'audio_feature_dim': snap_feature,
                    'norm': 'LN',
                    'middle_dim': 128,
                    'preprocess_dim': 16,
                    'cls_label_dim': 3,
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
        data_arg = {'dataset': 'nn.dataset.feeder_heatmapv2_meta.SubseqFeeder',
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
                     'noise_level': [0.3, 0.3],
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


def train_word2vec_skipgramv1(setting_name='word2vec_skipgramv1'):
    """
    simple_acganv2
    no sigmoid
    """
    random_seed = 404
    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': 'nn.net.word2vec_embed.Word2VecEmbed',
            'vocab_size': 65536,
            'feature_dim': 16,
            'train': True,
        }
        optimizer_arg = {
            'optimizer_type': 'SGD',
            'lr': lr,
            'momentum': 0.9,
            'nesterov': True,
            'weight_decay': 0.0001,
        }
        scheduler_arg = {
            'scheduler_type': 'MultiStepLR',
            'milestones': [50],
            'gamma': 1.
        }
        data_arg = {'dataset': 'nn.dataset.feeder_word2vec_skipgram.SkipGramFeeder',
                    'train_dataset_arg':
                        {
                            'label_idx_filepath': r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/all_label_idx.pkl',
                            'window_size': 5,
                            'neg_samples_per_center': 5,
                            'random_seed': random_seed,
                            'beat_divisor': 8,
                            'discard_hparam': 0.0001,
                            'neg_sampling_hparam': 0.75,
                            'take_first': None,
                         },
                    'batch_size': 256 * 8,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {
            'loss_type': 'binary_cross_entropy_with_logits',
            'params': {},
        }
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_dir': '/home/data1/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_step': 1,
                      'model_save_step_per_epoch': 8,
                      'train_state_save_step': 1,
                      }
        train_arg = {'epoch': 16,
                     'start_epoch': None,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'skipgram',
                       'num_classes': 5,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack_retain_type',
                       'grad_alter_fn': None,
                       # 'grad_alter_fn': 'norm',
                       # 'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)


def train_acgan_embeddingv1(setting_name='acgan_embeddingv1'):
    """
    acganv1_embedding
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
            'model_type': ['nn.net.acgan_embeddingv1.Generator', 'nn.net.acgan_embeddingv1.Discriminator'],
            'params': [
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'noise_dim': 16,
                    'norm': 'LN',
                    'middle_dim': 128,
                    'preprocess_dim': 32,
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
        data_arg = {'dataset': 'nn.dataset.feeder_embedding.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'/home/data1/xiezheng/osu_mapper/preprocessed_v5',
                         'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': 10240,
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': rnd_bank_size,
                         'level_coeff': 0.5,
                         },
                    'batch_size': 256,
                    'shuffle': False,
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
                     # 'noise_level_step': [(50 + 25 * i) for i in range(9)],
                     # 'noise_level': [(1 - 0.1 * i) for i in range(10)],
                     # 'noise_level_step': [30, 60, 90, 120],
                     # 'noise_level': [0.5, 0.5, 0.5, 0.5, 0.5],
                     'noise_level_step': [1,],
                     'noise_level': [0.5, 0.5],
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
                       'output_device': 0,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'acwganwb',
                       'num_classes': 5,
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack',
                       # 'grad_alter_fn': None,
                       'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 10},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)


def train_acgan_embeddingv3(setting_name='acgan_embeddingv3'):
    """
    acganv3_embedding
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
            'model_type': ['nn.net.acgan_embeddingv3.Generator', 'nn.net.acgan_embeddingv3.Discriminator'],
            'params': [
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'noise_dim': 16,
                    'norm': 'LN',
                    'middle_dim': 128,
                    'preprocess_dim': 32,
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
        data_arg = {'dataset': 'nn.dataset.feeder_embedding.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v5',
                         'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': 10240,
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': rnd_bank_size,
                         'level_coeff': 0.5,
                         },
                    'batch_size': 256,
                    'shuffle': False,
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
                     'noise_level_step': [1,],
                     'noise_level': [1.0, 1.0],
                     # 'noise_level': [0.5, 0.3, 0.1, 0.05, 0.],
                     # 'lambda_gp': 10,
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
                       'output_device': 0,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'train_type': 'acganwb',
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


def train_acgan_embeddingv4(setting_name='acgan_embeddingv4'):
    """
    acganv1_embedding
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
            'model_type': ['nn.net.acgan_embeddingv4.Generator', 'nn.net.acgan_embeddingv4.Discriminator'],
            'params': [
                {
                    'n_snap': subseq_snaps,
                    'tgt_embedding_dim': 16,
                    'tgt_coord_dim': 2,
                    'audio_feature_dim': mel_features,
                    'noise_dim': 16,
                    'norm': 'LN',
                    'middle_dim': 128,
                    'preprocess_dim': 32,
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
                {'milestones': [50], 'gamma': 1.},
                {'milestones': [50], 'gamma': 1.},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.feeder_embedding.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v5',
                         'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': 10240,
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': rnd_bank_size,
                         'level_coeff': 0.5,
                         },
                    'batch_size': 256,
                    'shuffle': False,
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
                     # 'noise_level_step': [(50 + 25 * i) for i in range(9)],
                     # 'noise_level': [(1 - 0.1 * i) for i in range(10)],
                     # 'noise_level_step': [30, 60, 90, 120],
                     # 'noise_level': [0.5, 0.5, 0.5, 0.5, 0.5],
                     'noise_level_step': [1,],
                     'noise_level': [0.5, 0.5],
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
                       'output_device': 0,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
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


def train_acgan_embeddingv5(setting_name='acgan_embeddingv5'):
    """
    acganv1_embedding
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
            'model_type': ['nn.net.acgan_embeddingv5.Generator', 'nn.net.acgan_embeddingv5.Discriminator'],
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
                    'audio_preprocess_dim': 4,
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
                {'milestones': [50], 'gamma': 1.},
                {'milestones': [50], 'gamma': 1.},
            ]
        }
        data_arg = {'dataset': 'nn.dataset.feeder_embedding.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v5',
                         'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         'take_first': 10240,
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': rnd_bank_size,
                         'level_coeff': 0.5,
                         },
                    'batch_size': 256,
                    'shuffle': False,
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
                     # 'noise_level_step': [(50 + 25 * i) for i in range(9)],
                     # 'noise_level': [(1 - 0.1 * i) for i in range(10)],
                     # 'noise_level_step': [30, 60, 90, 120],
                     # 'noise_level': [0.5, 0.5, 0.5, 0.5, 0.5],
                     'noise_level_step': [1,],
                     'noise_level': [0.5, 0.5],
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
                       'output_device': 0,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
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


def test_gen_capacity(setting_name='test_gan_capacity'):
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


def train_acgan_embeddingv6_noise(setting_name='acgan_embeddingv6_noise'):
    """
    acganv1_embedding
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

    for gen_lr, dis_lr in [[0.00003, 0.00003]]:
        print('init lr %s' % str(gen_lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': ['nn.net.acgan_embeddingv6.Generator', 'nn.net.acgan_embeddingv6.Discriminator'],
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
                     'noise_level_step': [60, 120, 180, 240, 300],
                     'noise_level': [1, 0.8, 0.6, 0.4, 0.2, 0.],
                     # 'noise_level': [10, 10, 5, 5, 5],
                     # 'noise_level': [5, 2.5, 1, 0.5, 0.25],
                     # 'noise_level': [0.5, 0.3, 0.1, 0.05, 0.],
                     'period': 15,
                     'lambda_gp': 500,
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
                       'output_device': 0,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
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
                       'grad_alter_fn_arg': {'max_norm': 5},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)


if __name__ == '__main__':
    # train_simple_acganv1_heatmapv2('simple_acganv1_heatmapv2_20230818_g0.0001_d0.0001')
    # train_simple_acganv1_heatmapv2('simple_acganv1_heatmapv2_20230821_g0.00003_d0.00003_rev1')
    # train_simple_acganv2_heatmapv2('simple_acganv2_heatmapv2_20230823_g0.0001_d0.0001_constlr')
    # train_simple_acganv3_heatmapv3('simple_acganv3_heatmapv3_20230829_g0.00001_d0.00001_constlr')
    # train_simple_acganv3_heatmapv3('simple_acganv3_heatmapv3_20230830_g0.000005_d0.000005_constlr')
    # setting_name = 'seq2seq_lr0.1'
    # train_seq2seq(setting_name)
    # setting_name = 'rnnv3_nolabel_lr0.1'
    # train_rnnv3_nolabel(setting_name)
    # setting_name = 'mlpv2_nolabel_lr0.1'
    # train_mlpv2_nolabel(setting_name)
    # train_word2vec_skipgramv1('word2vec_skipgramv1_0.1_constlr_dim_16_early_stop')
    # train_acgan_embeddingv1('acgan_embeddingv1_20231012_g0.00001_d0.00001_grad_clip_norm10')
    # train_acgan_embeddingv3('acgan_embeddingv3_20231016_g0.00001_d0.00001_grad_clip_norm1')
    # train_acgan_embeddingv5('acgan_embeddingv5_wgangp_20231017_g0.00001_d0.00001_grad_clip_norm5')
    # test_gen_capacity('20231021_acgan_embeddingv5_capacity_lr0.00001')
    # train_acgan_embeddingv5_noise('20231024_acgan_embeddingv15_glr0.00001_dlr0.0001_5_dp0.33_cls100_batch_noise_sched_filter_shuffle')
    # train_acgan_embeddingv6_noise('20231026_acgan_embeddingv6_glr0.00003_dlr0.00003_1_5_dp1_cls10_gl500_small_kernel_shift_embedding')

    # train_acgan_embeddingv6_noise('20231026_acgan_embeddingv6_glr0.00003_dlr0.00003_1_5_dp1_cls10_gl500_small_kernel_shift_embedding')
    # acgan_embedding.train_acgan_embeddingv6_only_coord('20231027_acgan_embeddingv6_only_coord_gp_lp10')
    # acgan_embedding.train_acgan_embeddingv6_only_embedding('20231027_acgan_embeddingv6_only_embedding_gp_lp10')
    # acgan_embedding.train_acgan_embeddingv7_only_coord('20231101_acgan_embeddingv7_only_coord_gp_lp10_lr0.00003_no_input_norm')
    # acgan_embedding.train_acgan_embeddingv7_only_embedding('20231101_acgan_embeddingv7_only_embedding_gp_lp10_lr0.00003_no_input_norm')
    # acgan_embedding.train_acgan_embeddingv7('20231103_acgan_embeddingv7_gp_lp10_glr0.0001_dlr0.00003_coord_noise_lb0.3_cls25')
    # acgan_embedding.train_acgan_embeddingv8('20231107_acgan_embeddingv8_gp_lp10_glr0.00001_dlr0.00001_new_label')
    # test_gen_capacity('20231108_acgan_embeddingv8_capacity_lr0.00001')
    # acgan_embedding.train_acgan_embeddingv9('20231111_acgan_embeddingv9_gp_lp10_glr1e-4_dlr1e-4_more_layers2')
    # acgan_embedding.train_acgan_embeddingv10('20231113_acgan_embeddingv10_gp_lp10_glr7e-5_dlr7e-5')
    # cddpm.train_cddpm('20230118_cddpm_0.001_datasetv6')
    # cddpm.train_cddpm('20230118_cddpm_0.0001_datasetv6_start_pos_meta')
    train_script_cddpm.train_cddpm_datasetv7('20230127_cddpm_0.0003_datasetv7_sched')
