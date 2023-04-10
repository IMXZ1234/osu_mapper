import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from system.base_sys import Train
from util.general_util import recursive_wrap_data


class TrainVAE(Train):
    def run_train(self):
        self.init_train_state()
        self.teacher_forcing_ratio = self.config_dict['train_arg']['init_teacher_forcing_ratio']
        epoch_loss_list = []

        for epoch in range(self.epoch):
            epoch_loss = self.train_epoch(epoch)
            epoch_loss_list.append(epoch_loss)

            self.save_model(epoch)

        plt.plot(np.arange(len(epoch_loss_list)), epoch_loss_list)
        plt.show()

    def train_epoch(self, epoch):
        # linear scheduled teacher_forcing_ratio
        cur_teacher_forcing_ratio = self.teacher_forcing_ratio * (1 - epoch / self.epoch)
        # loss_G, loss_D = self.loss
        total_loss = 0
        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            self.optimizer.zero_grad()
            # print(real_gen_output)

            # batch_size = cond_data.shape[0]
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            recon, mu, log_std = self.model(cond_data, real_gen_output, teacher_forcing_ratio=cur_teacher_forcing_ratio)
            recon_loss, kl_loss = self.model.loss_function(recon, mu, log_std, real_gen_output)

            # kl_loss *= 0.025
            # kl annealing
            kl_loss = kl_loss * (epoch / self.epoch)
            print('annealed kl_loss %.4f' % kl_loss)
            loss = recon_loss + kl_loss
            total_loss += loss.item()

            loss.backward()
            if self.grad_alter_fn is not None:
                self.grad_alter_fn(self.model.parameters())
            self.optimizer.step()
        avg_loss = total_loss / len(self.train_iter.dataset)
        print('avg_loss %d' % avg_loss)
        recon = recon[0].detach().cpu().numpy()
        print(np.where(recon[:, 0] > 0.5, 1, 0))
        print(np.where(recon[:, 1] > 0.5, 2, 0))
        # print(sampled[:, :2][:32])
        print(recon[:, 2:][:32])
        sampled = self.model.sample(cond_data)[0].detach().cpu().numpy()
        print(np.where(sampled[:, 0] > 0.5, 1, 0))
        print(np.where(sampled[:, 1] > 0.5, 2, 0))
        # print(sampled[:, :2][:32])
        print(sampled[:, 2:][:32])
        # self.scheduler.step()
        return avg_loss