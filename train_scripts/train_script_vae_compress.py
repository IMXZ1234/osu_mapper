import sys
import os
sys.path.append(
    os.path.abspath('.')
)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import yaml

from nn.train import train_with_config


def train_vae_compress(setting_name='vae_compress'):
    """
    acganv1_embedding
    """
    random_seed = 404
    # 32 beat per subseq
    beat_divisor = 8
    mel_frame_per_snap = 8
    subseq_beats = 32
    subseq_snaps = subseq_beats * beat_divisor
    embedding_size = subseq_beats * 16
    label_size = 2 * subseq_snaps
    rnd_bank_size = max(embedding_size, label_size) * 1024
    mel_features = 40
    meta_features = 3
    # NA, circle, slider, spinner
    num_classes = 4

    batch_size = 1024
    bucket_size = batch_size // 16

    latent_seq_len = subseq_snaps // 16

    for lr in [0.000001]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/train/%s.yaml' % setting_name
        model_arg = {
            'model_type': 'nn.net.vae_compress.VAE',
            'n_tokens': 4,
            'hidden_dim': 16,
            'discrete_in_channels': 2,
            'continuous_in_channels': 2,
            'enc_stride_list': [2, 2, 2, 2],
            'enc_out_channels_list': [8, 16, 32, 64],
            'dec_stride_list': [1, 1, 2,
                                1, 1, 2,
                                1, 1, 2,
                                1, 1, 2,
                                1, 1,],
            'dec_out_channels_list': [128, 128, 64,
                                      64, 64, 64,
                                      64, 64, 32,
                                      32, 32, 32,
                                      32, 6,],
        }
        optimizer_arg = {
            'optimizer_type': 'Adam',
            'lr': lr,
        }
        scheduler_arg = {
            'scheduler_type': 'MultiStepLR',
            'milestones': [64],
            'gamma': 0.1,
        }
        data_arg = {'dataset': 'nn.dataset.feeder_labelv7.SubseqFeeder',
                    'train_dataset_arg':
                        {'save_dir': r'/home/xiezheng/data/preprocessed_v7',
                         # 'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                         'subseq_snaps': subseq_snaps,
                         'random_seed': random_seed,
                         'use_random_iter': True,
                         # totally 52795 samples
                         'take_range': None,
                         # 'take_range': [0, 1024],
                         'pad': False,
                         'beat_divisor': beat_divisor,
                         'rnd_bank_size': None,
                         'level_coeff': 0,
                         'item': 'coord_type',
                         'cache': batch_size,
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
                      'log_batch_step': 320,
                      'log_epoch_step': 1,
                      }
        train_arg = {'epoch': 256,
                     'eval_step': 1,
                     'start_epoch': 0,
                     }
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,

                       'latent_seq_len': latent_seq_len,
                       'kl_loss_anneal': True,
                       'kl_loss_lambda': 0.000005,
                       'continuous_loss_lambda': 15.,
                       'do_coord': False,
                       'num_classes': num_classes,
                       'output_device': 0,
                       # 'data_parallel_devices': [0, 1, 2, 3, 4, 5, 6, 7],
                       # 'weights_path': [
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_0_epoch_-1.pt',
                       #     r'resources/result/cganv8_with_batch_20221220/1/model_1_epoch_-1.pt',
                       # ],
                       'log_item': 'coord_type',
                       'train_type': 'vae_compress',
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
    train_vae_compress(
        r'20240319_vae_compress_encl4_decl14_incc_kl5e-6_cl15_e256'
        # r'20240316_vae_compress_l8incc_kl5e-6_cl25_e256'
        # r'20240316_vae_compress_l8incc_kl5e-6_cl25'
        # r'20240316_vae_compress_l8c16_kl5e-6anneal_cl25'
    )
