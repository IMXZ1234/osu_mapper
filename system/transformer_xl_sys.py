import collections
import math
import os
import pickle
from functools import wraps

import numpy as np
import torch
from einops import repeat, reduce
from torch import sqrt
from torch import nn
from torch.nn import functional as F
from torch.special import expm1
import random
from tqdm import tqdm

from system.base_sys import Train
from util.general_util import recursive_wrap_data
from util.plt_util import plot_signal, plot_multiple_signal


def init_weight(weight, init='normal',
                init_range=0.1,
                init_std=0.02, ):
    if init == 'uniform':
        nn.init.uniform_(weight, -init_range, init_range)
    elif init == 'normal':
        nn.init.normal_(weight, 0.0, init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m, proj_init_std=0.01,
                 init_std=0.02, ):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


class TransformerXLSYS(Train):
    def __init__(self,
                 config_dict,
                 task_type='classification',
                 **kwargs):
        super().__init__(config_dict, task_type, **kwargs)
        self.loss_lambda = self.config_dict['loss_lambda']
        self.do_coord = self.config_dict['do_coord']
        self.num_classes = self.config_dict['num_classes']
        self.num_classes_decoded = self.config_dict.get('num_classes_decoded', self.num_classes)
        self.decoder_filepath = self.config_dict.get('decoder_filepath', None)
        if self.decoder_filepath is not None:
            with open(self.decoder_filepath, 'rb') as f:
                self.decoder = pickle.load(f)
        else:
            self.decoder = {i: (i,) for i in range(self.num_classes)}

    def run_train(self):
        print('start training transformer_xl')
        rand_inst = random.Random(404)
        self.init_train_state()
        for epoch in range(self.start_epoch, self.epoch):
            self.logger.info("train_epoch %d", epoch)
            epoch_train_loss_list = []
            epoch_test_loss_list = []

            bar = tqdm(enumerate(self.train_iter), total=len(self.train_iter), ncols=80, desc='batch')
            for batch_idx, batch in bar:
                # batch: [context, inp_discrete, inp_continuous, tgt_discrete, tgt_continuous]
                # context: [N, n_mel, L', n_step] audio data,
                # inp: [N, 5, L, n_step] hit_type and cursor coord,
                # tgt: [N, L, n_step], [N, 2, L, n_step] hit_type and cursor coord (one step later),
                mems = []
                num_step = batch[0].shape[3]
                # step_bar = tqdm(range(num_step), total=num_step, ncols=80, desc='step')
                # for step_idx in step_bar:
                for step_idx in range(num_step):
                    step_batch = [item[..., step_idx] for item in batch]
                    step_batch = recursive_wrap_data(step_batch, self.output_device)
                    context, inp_discrete, inp_continuous, tgt_discrete, tgt_continuous = step_batch
                    # out: [N, 3, L], [N, 2, L] hit_type and cursor coord (one step later),
                    out_discrete, out_continuous, mems = self.model(
                        inp_discrete,
                        inp_continuous if self.do_coord else None,
                        context,
                        mems,
                    )
                    if self.do_coord:
                        loss_continuous = F.mse_loss(out_continuous, tgt_continuous)
                    else:
                        loss_continuous = 0
                    loss_discrete = F.cross_entropy(out_discrete, tgt_discrete)

                    loss = loss_discrete + loss_continuous * self.loss_lambda

                    bar.set_postfix(collections.OrderedDict({'train loss': loss.item()}))
                    epoch_train_loss_list.append(loss.item())

                    if self.grad_alter_fn is not None:
                        self.grad_alter_fn(self.model if self.grad_alter_fn_inp_type == 'model' else self.model.parameters(),
                                           **self.grad_alter_fn_arg)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.logger.info("batch train loss %.4f", loss)

                if (batch_idx+1) % self.model_save_batch_step == 0:
                    self.save_model(epoch, batch_idx)

                # generate some samples, use the last test batch as input
                if (batch_idx+1) % self.log_batch_step == 0:
                    self.logger.info("sample_batch %d", batch_idx)
                    self.model.eval()
                    with torch.no_grad():
                        # batch: [context, inp_discrete, inp_continuous, tgt_discrete, tgt_continuous]
                        # context: [N, n_mel, L', n_step] audio data,
                        # inp: [N, 5, L, n_step] hit_type and cursor coord,
                        # tgt: [N, L, n_step], [N, 2, L, n_step] hit_type and cursor coord (one step later),
                        # only run sampling for part of the test batch
                        num_gen_vis = 4
                        batch = [item[:num_gen_vis] for item in batch]
                        mems = []
                        num_step = batch[0].shape[3]
                        all_step_out_discrete, all_step_out_continuous = [], []
                        all_step_context, all_step_inp_discrete, all_step_inp_continuous, _, _ = batch
                        # first step token from data
                        out_last_discrete = recursive_wrap_data(all_step_inp_discrete[..., :1, 0], self.output_device)
                        out_last_continuous = recursive_wrap_data(all_step_inp_continuous[..., :1, 0], self.output_device)
                        for step_idx in tqdm(range(num_step), total=num_step, ncols=80, desc='step'):
                            context = recursive_wrap_data(all_step_context[..., step_idx], self.output_device)
                            # note here out_discrete is token_id after sampling [N, L]
                            out_discrete, out_continuous, mems, out_last_discrete, out_last_continuous = self.model.forward_segment(
                                out_last_discrete,
                                out_last_continuous if self.do_coord else None,
                                context,
                                mems,
                                rand_inst=rand_inst,
                            )
                            all_step_out_discrete.append(out_discrete)
                            all_step_out_continuous.append(out_continuous)
                        all_step_out_discrete = torch.cat(all_step_out_discrete, dim=1)
                        if self.do_coord:
                            all_step_out_continuous = torch.cat(all_step_out_continuous, dim=1)
                        # flatten steps
                        all_step_inp_discrete = all_step_inp_discrete.reshape(list(all_step_inp_discrete.shape[:-2]) + [-1])
                        all_step_inp_continuous = all_step_inp_continuous.reshape(list(all_step_inp_continuous.shape[:-2]) + [-1])
                        self.log_gen_output(all_step_out_discrete, all_step_out_continuous, epoch, batch_idx, 'output_')
                        self.log_gen_output(all_step_inp_discrete, all_step_inp_continuous, epoch, batch_idx, 'input_')
                    self.model.train()

            self.scheduler.step()
            self.logger.info("epoch train loss %.4f", sum(epoch_train_loss_list) / len(epoch_train_loss_list))

            self.save_model(epoch)
            self.save_train_state(self.model, self.optimizer, epoch)

            if (epoch+1) % self.eval_step == 0:
                self.logger.info("eval_epoch %d", epoch)
                self.model.eval()

                # calculate loss on test dataset
                bar = tqdm(enumerate(self.test_iter), total=len(self.test_iter), ncols=80, desc='batch')
                for batch_idx, batch in bar:
                    # batch: [context, inp_discrete, inp_continuous, tgt_discrete, tgt_continuous]
                    # context: [N, n_mel, L', n_step] audio data,
                    # inp: [N, 5, L, n_step] hit_type and cursor coord,
                    # tgt: [N, L, n_step], [N, 2, L, n_step] hit_type and cursor coord (one step later),
                    mems = []
                    num_step = batch[0].shape[3]
                    # step_bar = tqdm(range(num_step), total=num_step, ncols=80, desc='step')
                    # for step_idx in step_bar:
                    for step_idx in range(num_step):
                        step_batch = [item[..., step_idx] for item in batch]
                        step_batch = recursive_wrap_data(step_batch, self.output_device)
                        context, inp_discrete, inp_continuous, tgt_discrete, tgt_continuous = step_batch
                        # out: [N, 3, L], [N, 2, L] hit_type and cursor coord (one step later),
                        out_discrete, out_continuous, mems = self.model(
                            inp_discrete,
                            inp_continuous if self.do_coord else None,
                            context,
                            mems,
                        )
                        if self.do_coord:
                            loss_continuous = F.mse_loss(out_continuous, tgt_continuous)
                        else:
                            loss_continuous = 0
                        loss_discrete = F.cross_entropy(out_discrete, tgt_discrete)

                        loss = loss_discrete + loss_continuous * self.loss_lambda

                        bar.set_postfix(collections.OrderedDict({'eval loss': loss.item()}))
                        epoch_test_loss_list.append(loss.item())

                    self.logger.info("batch test loss %.4f", loss)

                self.logger.info("epoch test loss %.4f", sum(epoch_test_loss_list) / len(epoch_test_loss_list))
                self.model.train()

        self.save_model(-1)
        self.save_train_state(self.model, self.optimizer, -1)

    def decode(self, out_discrete):
        """
        out_discrete: int ndarray [..., L] -> [..., L*label_group_size]
        """
        old_shape = out_discrete.shape[:-1]
        flattened = out_discrete.reshape(-1)
        flattened_decoded = np.array([self.decoder[i] for i in flattened])
        return flattened_decoded.reshape(list(old_shape) + [-1])

    def log_gen_output(self, out_discrete, out_continuous, epoch, batch, prefix=''):
        """
        out_discrete: N, L, torch tensor
        """
        tgt_subdir = '%d_%d' % (epoch, batch)
        save_dir = os.path.join(self.log_dir, tgt_subdir)
        os.makedirs(save_dir, exist_ok=True)

        # run decoding
        out_discrete = self.decode(out_discrete.detach().cpu().numpy())
        N, L = out_discrete.shape
        # N, L -> N, L, C one hot
        ho_type_one_hot = np.eye(self.num_classes_decoded)[out_discrete.reshape(-1)].reshape([N, L, self.num_classes_decoded])
        # N, L, C -> N, C, L
        ho_type_one_hot = np.transpose(ho_type_one_hot, [0, 2, 1])
        ho_type_one_hot = ho_type_one_hot[:, 1:]
        # circle
        ho_type_one_hot[:, 0] *= 3
        # slider
        ho_type_one_hot[:, 1] *= 2
        for sample_idx in range(out_discrete.shape[0]):
            plot_multiple_signal(ho_type_one_hot[sample_idx],
                                 'ho',
                                 os.path.join(save_dir, prefix + '%d_ho.jpg' % sample_idx),
                                 show=False)

    def sample_start_token(self, first_segment_context):
        pass


# def sample(model):
#     # Turn on evaluation mode which disables dropout.
#     self.model.eval()
#
#     # If the model does not use memory at all, make the ext_len longer.
#     # Otherwise, make the mem_len longer and keep the ext_len the same.
#     if args.mem_len == 0:
#         model.reset_length(args.eval_tgt_len,
#             args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
#     else:
#         model.reset_length(args.eval_tgt_len,
#             args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)
#
#     # Evaluation
#     total_len, total_loss = 0, 0.
#     with torch.no_grad():
#         mems = tuple()
#         for i, (data, target, seq_len) in enumerate(eval_iter):
#             if args.max_eval_steps > 0 and i >= args.max_eval_steps:
#                 break
#             ret = model(data, target, *mems)
#             loss, mems = ret[0], ret[1:]
#             loss = loss.mean()
#             total_loss += seq_len * loss.float().item()
#             total_len += seq_len
#
#     # Switch back to the training mode
#     model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
#     model.train()
#
# return total_loss / total_len
