import collections
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from system.base_sys import Train
from util.general_util import recursive_wrap_data
from util.plt_util import plot_signal, plot_multiple_signal


class TrainVAECompress(Train):
    def __init__(self, config_dict,
                 task_type: str = 'classification',
                 **kwargs):
        super(TrainVAECompress, self).__init__(config_dict, task_type)
        self.kl_loss_lambda = self.config_dict['kl_loss_lambda']
        self.continuous_loss_lambda = self.config_dict['continuous_loss_lambda']
        self.latent_seq_len = self.config_dict['latent_seq_len']
        self.kl_loss_anneal = self.config_dict['kl_loss_anneal']

    def run_train(self):
        self.init_train_state()
        epoch_loss_list = []

        for epoch in range(self.epoch):
            epoch_loss = self.train_epoch(epoch)
            epoch_loss_list.append(epoch_loss)

            self.save_model(epoch)

        plt.figure()
        plt.plot(np.arange(len(epoch_loss_list)), epoch_loss_list)
        plt.savefig(os.path.join(self.log_dir, 'loss.jpg'))

    def train_epoch(self, epoch):
        total_loss = 0
        bar = tqdm(enumerate(self.train_iter), total=len(self.train_iter), ncols=120, desc='batch')
        for batch_idx, batch in bar:
            # batch_size = cond_data.shape[0]
            batch = recursive_wrap_data(batch, self.output_device)
            recon, mu, log_std = self.model(batch)
            recon_loss, kl_loss = self.model.loss_function(recon, mu, log_std, batch)

            discrete_loss, continuous_loss = recon_loss

            # kl_loss *= 0.025
            # kl annealing
            if self.kl_loss_anneal:
                kl_loss = kl_loss * (epoch / self.epoch)
            # print('annealed kl_loss %.4f' % kl_loss)
            bar.set_postfix(collections.OrderedDict({
                'dl': discrete_loss.item(),
                'cl': continuous_loss.item(),
                'kl': kl_loss.item()
            }))
            self.writer.add_scalar('dl', discrete_loss.item())
            self.writer.add_scalar('cl', continuous_loss.item())
            self.writer.add_scalar('kl', kl_loss.item())
            loss = discrete_loss + \
                   self.continuous_loss_lambda * continuous_loss + \
                   self.kl_loss_lambda * kl_loss
            total_loss += loss.item()
            self.writer.add_scalar('ttl', loss.item())
            self.writer.add_scalar('epoch', epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'])

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_alter_fn is not None:
                self.grad_alter_fn(self.model.parameters())
            self.optimizer.step()

            if (batch_idx + 1) % self.log_batch_step == 0:
                plot_first = 3
                self.logger.info("sample_batch %d", batch_idx)
                self.log_gen_output(batch, epoch, batch_idx, 'inp_', plot_first=plot_first, is_decoded=True)
                self.log_gen_output(recon, epoch, batch_idx, 'out_', plot_first=plot_first, is_decoded=False)

                # run sampling
                self.model.eval()
                with torch.no_grad():
                    sampled = self.model.sample(plot_first, self.latent_seq_len)
                self.model.train()
                self.log_gen_output(sampled, epoch, batch_idx, 'out_sample_', plot_first=plot_first, is_decoded=False)

        avg_loss = total_loss / len(self.train_iter.dataset)

        self.logger.info("epoch train loss %.4f", avg_loss)
        return avg_loss

    def log_gen_output(self, recon, epoch, batch_idx, prefix='gen', plot_first=3, is_decoded=True):
        output_dir = os.path.join(self.log_dir, 'output', 'e%d_b%d' % (epoch, batch_idx))
        os.makedirs(output_dir, exist_ok=True)

        # out_discrete: (if not is_decoded)[N, Cd, L] probability distribution, (if is_decoded)[N, L]
        # out_continuous: [N, C, L]
        embedding_output, coord_output = recon
        embedding_output = embedding_output.detach().cpu().numpy()
        coord_output = coord_output.detach().cpu().numpy()

        if is_decoded:
            N, L = embedding_output.shape
            all_sample_decoded = embedding_output
        else:
            N, n_tokens, L = embedding_output.shape
            # N, L
            all_sample_decoded = np.argmax(embedding_output, axis=1)

        # to one hot -> [batch_size, n_tokens, n_snaps]
        embedding_output = np.transpose(np.eye(4)[all_sample_decoded.reshape(-1)].reshape([N, L, -1]), [0, 2, 1])[:, 1:]
        # circle
        embedding_output[:, 0] *= 3
        # slider
        embedding_output[:, 1] *= 2

        with open(os.path.join(output_dir, prefix + r'hit_signal.txt'), 'w') as f:
            for sample_decoded in all_sample_decoded:
                f.write(str(sample_decoded.tolist()) + '\n')
        with open(os.path.join(output_dir, prefix + r'cursor_signal.txt'), 'w') as f:
            for sample_coord_output in coord_output:
                # [n_snaps, coord_num]
                f.write(str(sample_coord_output.T.tolist()) + '\n')
        for i, (sample_coord_output, sample_embedding_output) in enumerate(zip(coord_output, embedding_output)):
            if i >= plot_first:
                break

            fig_array, fig = plot_multiple_signal(sample_embedding_output,
                                 'ho',
                                 os.path.join(output_dir, prefix + '%d_ho.jpg' % i),
                                 show=False, signal_len_per_inch=50)
            self.writer.add_figure(prefix + '%d_ho.jpg' % i, fig, epoch)

            fig_array, fig = plot_multiple_signal(sample_coord_output,
                                 'coord',
                                 os.path.join(output_dir, prefix + '%d_coord.jpg' % i),
                                 show=False, signal_len_per_inch=50)
            self.writer.add_figure(prefix + '%d_coord.jpg' % i, fig, epoch)
