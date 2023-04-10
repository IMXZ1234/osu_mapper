import sys

import torch
from torch.nn import functional as F
from tqdm import tqdm

from system.base_sys import Train
from util.general_util import recursive_wrap_data


class TrainSeq2Seq(Train):
    def run_train(self):
        self.init_train_state()

        for epoch in range(self.epoch):
            self.train_epoch(epoch)

            self.save_model(epoch)

            self.test_epoch(epoch)

    def test_epoch(self, epoch):
        pass
        # for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.test_iter)):
        #     cond_data = recursive_wrap_data(cond_data, self.output_device)
        #     real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)

    def train_epoch(self, epoch):
        # loss_G, loss_D = self.loss
        teacher_forcing_ratio = self.train_extra['teacher_forcing_ratio'] if 'teacher_forcing_ratio' in self.train_extra else 0.5
        total_loss = 0
        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            self.optimizer.zero_grad()
            # print(real_gen_output)

            batch_size, seq_len, _ = cond_data.shape
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            out = self.model(cond_data, real_gen_output, teacher_forcing_ratio)

            pred_type_label, pred_ho_pos = \
                out[:, :, :3].reshape([batch_size * seq_len, -1]), \
                out[:, :, 3:]
            type_label, ho_pos = \
                real_gen_output[:, :, 0].reshape(-1).long(), \
                real_gen_output[:, :, 1:]

            # print('type_label')
            # print(type_label.shape)
            # print('pred_type_label')
            # print(pred_type_label.shape)
            # print('pred_ho_pos')
            # print(pred_ho_pos.shape)
            # print('ho_pos')
            # print(ho_pos.shape)
            type_label_loss = F.cross_entropy(pred_type_label, type_label, weight=torch.tensor([1., 10., 1.], device=cond_data.device))
            ho_pos_loss = F.mse_loss(pred_ho_pos, ho_pos)
            loss = type_label_loss + ho_pos_loss

            total_loss += loss.item()

            loss.backward()
            if self.grad_alter_fn is not None:
                self.grad_alter_fn(self.model.parameters())
            self.optimizer.step()
        print('avg_loss %.4f' % (total_loss / len(self.train_iter.dataset)))
        print(torch.argmax(pred_type_label, dim=-1).detach().cpu().numpy())
        print(pred_ho_pos.detach().cpu().numpy())
        # self.scheduler.step()