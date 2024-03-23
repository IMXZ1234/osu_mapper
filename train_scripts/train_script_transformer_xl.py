import sys
import os
sys.path.append(
    os.path.abspath('.')
)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import yaml

from nn.train import train_with_config


def train_transformer_xl(setting_name='transformer_xl'):
    """
    acganv1_embedding
    """
    random_seed = 404
    # 32 beat per subseq
    beat_divisor = 8
    mel_frame_per_snap = 8
    subseq_beats = 16
    subseq_snaps = subseq_beats * beat_divisor
    embedding_size = subseq_beats * 16
    label_size = 2 * subseq_snaps
    rnd_bank_size = max(embedding_size, label_size) * 1024
    mel_features = 40
    meta_features = 3
    # NA, circle, slider, spinner
    num_classes = 4

    batch_size = 64
    bucket_size = batch_size

    for lr in [0.00003]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': 'nn.net.transformer_xl.mem_transformer.MemTransformerLM',
            'n_token': num_classes,
            'n_layer': 8,
            'n_head': 8,
            'd_model': 256,
            'd_head': 32,
            'd_inner': 256,
            'dropout': 0.1,
            'dropatt': 0,
            'tie_weight': True,
            # same as d_model
            'd_embed': None,
            'div_val': 1,
            'tie_projs': [False],
            'pre_lnorm': False,
            'tgt_len': subseq_snaps,
            'ext_len': 0,
            'mem_len': subseq_snaps,
            'cutoffs': [],
            'adapt_inp': False,
            'same_length': False,
            'attn_type': 0,
            'clamp_len': -1,
            'sample_softmax': -1,
            'context_compress': mel_frame_per_snap,
            'd_context': mel_features + meta_features,
            # do not do coord
            'd_continuous': 0,
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': 'Adam',
            'lr': lr,
        }
        scheduler_arg = {
            'scheduler_type': 'MultiStepLR',
            'milestones': [1],
            'gamma': 0.1,
        }
        data_arg = {'dataset': 'nn.dataset.feeder_v7_tranformer_xl.Feeder',
                    'train_sampler': 'nn.dataset.samplers.BucketBatchSampler',
                    'train_sampler_type': 'batch_sampler',
                    'train_sampler_arg': {
                        'batch_size': batch_size,
                        'rand_seed': random_seed,
                    },
                    'test_sampler': 'nn.dataset.samplers.BucketBatchSampler',
                    'test_sampler_type': 'batch_sampler',
                    'test_sampler_arg': {
                        'batch_size': batch_size,
                        'rand_seed': random_seed,
                    },
                    'train_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v7',
                         # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         # totally 52795 samples
                         'take_range': [0, 50000],
                         # 'take_range': [0, 1024],
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': None,
                         'level_coeff': 0,
                         'item': 'coord_type',
                         'cache': batch_size,
                         'bucket_size': bucket_size,  # num samples in a single bucket
                         },
                    'test_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v7',
                         # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         # 'take_range': [1024, 1536],
                         'take_range': [50000, 52796],
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': None,
                         'level_coeff': 0,
                         'item': 'coord_type',
                         'cache': batch_size,
                         'bucket_size': bucket_size,  # num samples in a single bucket
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 8,
                    'drop_last': False}
        loss_arg = {
            'loss_type': 'MSELoss',
        }
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': '/home/data/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_dir': '/home/data/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_step': 1,
                      'model_save_batch_step': 512,
                      'train_state_save_step': 1,
                      'log_batch_step': 64,
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

                       'loss_lambda': 1.,
                       'do_coord': False,
                       'num_classes': num_classes,
                       'output_device': 1,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'log_item': 'coord_type',
                       'train_type': 'transformer_xl',
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack_retain_type',
                       # 'grad_alter_fn': None,
                       'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 5},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)


def train_transformer_xl_label_group(setting_name='transformer_xl_label_group'):
    """
    acganv1_embedding
    """
    random_seed = 404
    # 32 beat per subseq
    beat_divisor = 8
    mel_frame_per_snap = 8
    subseq_beats = 16
    subseq_snaps = subseq_beats * beat_divisor
    embedding_size = subseq_beats * 16
    label_size = 2 * subseq_snaps
    rnd_bank_size = max(embedding_size, label_size) * 1024
    mel_features = 40
    meta_features = 3
    # NA, circle, slider, spinner
    num_classes_decoded = 4  # number of ho type
    num_classes = 71  # label_group_size=4
    # num_classes = 355  # label_group_size=8

    batch_size = 64
    bucket_size = batch_size
    label_group_size = 4

    for lr in [0.0001]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': 'nn.net.transformer_xl.mem_transformer.MemTransformerLM',
            'n_token': num_classes,
            'n_layer': 8,
            'n_head': 8,
            'd_model': 256,
            'd_head': 32,
            'd_inner': 256,
            'dropout': 0.1,
            'dropatt': 0,
            'tie_weight': True,
            # same as d_model
            'd_embed': None,
            'div_val': 1,
            'tie_projs': [False],
            'pre_lnorm': False,
            'tgt_len': subseq_snaps,
            'ext_len': 0,
            'mem_len': subseq_snaps,
            'cutoffs': [],
            'adapt_inp': False,
            'same_length': False,
            'attn_type': 0,
            'clamp_len': -1,
            'sample_softmax': -1,
            'context_compress': mel_frame_per_snap * label_group_size,
            'd_context': mel_features + meta_features,
            # do not do coord
            'd_continuous': 0,
        }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {
            'optimizer_type': 'Adam',
            'lr': lr,
        }
        scheduler_arg = {
            'scheduler_type': 'MultiStepLR',
            'milestones': [1, 3],
            'gamma': 0.1,
        }
        data_arg = {'dataset': 'nn.dataset.feeder_v7_tranformer_xl_label_group.Feeder',
                    'train_sampler': 'nn.dataset.samplers.BucketBatchSampler',
                    'train_sampler_type': 'batch_sampler',
                    'train_sampler_arg': {
                        'batch_size': batch_size,
                        'rand_seed': random_seed,
                    },
                    'test_sampler': 'nn.dataset.samplers.BucketBatchSampler',
                    'test_sampler_type': 'batch_sampler',
                    'test_sampler_arg': {
                        'batch_size': batch_size,
                        'rand_seed': random_seed,
                    },
                    'train_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v7',
                         # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                         'subseq_snaps': subseq_snaps,
                         'label_group_size': label_group_size,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         # totally 52795 samples
                         'take_range': [0, 50000],
                         # 'take_range': [0, 1024],
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': None,
                         'level_coeff': 0,
                         'item': 'coord_type',
                         'cache': batch_size,
                         'bucket_size': bucket_size,  # num samples in a single bucket
                         'take_beat_divisor': [2, 4, 8],
                         },
                    'test_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v7',
                         # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         # 'take_range': [1024, 1536],
                         'take_range': [50000, 52796],
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': None,
                         'level_coeff': 0,
                         'item': 'coord_type',
                         'cache': batch_size,
                         'bucket_size': bucket_size,  # num samples in a single bucket
                         },
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 8,
                    'drop_last': False}
        loss_arg = {
            'loss_type': 'MSELoss',
        }
        pred_arg = {'pred_type': 'nn.pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': '/home/data/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_dir': '/home/data/xiezheng/osu_mapper/result/' + setting_name,
                      'model_save_step': 1,
                      'model_save_batch_step': 512,
                      'train_state_save_step': 1,
                      'log_batch_step': 64,
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

                       'decoder_filepath': r'/home/xiezheng/data/preprocessed_v7/idx2labelgroup_size4.pkl',
                       'loss_lambda': 1.,
                       'do_coord': False,
                       'num_classes': num_classes,
                       'num_classes_decoded': num_classes_decoded,
                       'output_device': 1,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'log_item': 'coord_type',
                       'train_type': 'transformer_xl',
                       'random_seed': random_seed,
                       'collate_fn': 'nn.dataset.collate_fn.tensor_list_recursive_stack_retain_type',
                       # 'grad_alter_fn': None,
                       'grad_alter_fn': 'norm',
                       'grad_alter_fn_arg': {'max_norm': 5},
                       'cal_acc_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'nn.metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=None, format_config=False)


if __name__ == '__main__':
    train_transformer_xl_label_group(
        r'20240322_transformer_xl_label_group_size4_only_ho_type_offset_subseq16'
    )
