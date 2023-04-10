import os
import os
import random
import sys

import torch
import yaml
from tqdm import tqdm

from system.base_sys import Train
from util.general_util import recursive_wrap_data
from util.plt_util import plot_loss


class TrainGAN(Train):
    def run_train(self):
        self.init_train_state()
        control_dict = {'save_model_next_epoch': False}
        with open(self.control_file_path, 'w') as f:
            yaml.dump(control_dict, f)

        self.dis_loss_list, self.gen_loss_list = [], []

        # # GENERATOR MLE TRAINING
        # print('Starting Generator MLE Training...')
        # for epoch in range(self.config_dict['train_arg']['generator_pretrain_epoch']):
        #     print('epoch %d : ' % (epoch + 1), end='')
        #     self.train_generator_MLE()
        # self.save_model(-16)

        # torch.save(gen.state_dict(), pretrained_gen_path)
        # gen.load_state_dict(torch.load(pretrained_gen_path))

        # PRETRAIN DISCRIMINATOR
        self.logger.info('\nStarting Discriminator Training...')
        for epoch in range(self.config_dict['train_arg']['discriminator_pretrain_epoch']):
            self.logger.info('epoch %d : ' % (epoch + 1))
            self.train_discriminator()

        # torch.save(dis.state_dict(), pretrained_dis_path)
        # dis.load_state_dict(torch.load(pretrained_dis_path))

        self.run_adv_training()

        self.save_model(-1)
        log_dir = self.config_dict['output_arg']['log_dir']
        plot_loss(self.dis_loss_list, 'discriminator loss',
                  save_path=os.path.join(log_dir, 'discriminator_loss.png'), show=True)
        plot_loss(self.gen_loss_list, 'generator loss',
                  save_path=os.path.join(log_dir, 'generator_loss.png'), show=True)
        # self.save_properties()

    def run_adv_training(self):
        # # ADVERSARIAL TRAINING
        self.logger.info('\nStarting Adversarial Training...')
        acc_limit = self.config_dict['train_arg']['acc_limit'] if 'acc_limit' in self.config_dict['train_arg'] else 0.6

        for epoch in range(self.epoch):
            self.logger.info('\n--------\nEPOCH %d\n--------' % (epoch + 1))
            if self.config_dict['train_arg']['adaptive_adv_train']:
                while self.train_generator() > acc_limit:
                    pass
                while self.train_discriminator() < acc_limit:
                    pass
            else:
                # TRAIN GENERATOR
                self.logger.info('\nAdversarial Training Generator : ')
                for i in range(self.config_dict['train_arg']['adv_generator_epoch']):
                    self.train_generator()

                # TRAIN DISCRIMINATOR
                self.logger.info('\nAdversarial Training Discriminator : ')
                for i in range(self.config_dict['train_arg']['adv_discriminator_epoch']):
                    self.train_discriminator()
            if (epoch + 1) % self.model_save_step == 0:
                self.save_model(epoch, (0,))

    def train_generator(self):
        """
        The generator is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        optimizer_G, optimizer_D = self.optimizer[1], self.optimizer[2]
        # loss_G, loss_D = self.loss
        gen, dis = self.model
        epoch_dis_acc = 0
        epoch_pg_loss = 0
        total_sample_num = 0
        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            batch_size = cond_data.shape[0]
            total_sample_num += batch_size
            cond_data = recursive_wrap_data(cond_data, self.output_device)

            fake = gen(cond_data)
            dis_cls_out = dis(cond_data, fake)
            epoch_dis_acc += torch.sum(dis_cls_out < 0.5).data.item()

            optimizer_G.zero_grad()

            gen_loss = -torch.mean(dis_cls_out)

            epoch_pg_loss += gen_loss.item()
            gen_loss.backward()
            optimizer_G.step()

        epoch_dis_acc = epoch_dis_acc / total_sample_num
        self.logger.info('epoch_dis_acc %.3f' % epoch_dis_acc)

        sys.stdout.flush()
        # print('sample')
        sample = gen(cond_data)[0].cpu().detach().numpy()
        self.logger.info(sample[:])

        return epoch_dis_acc

    def train_discriminator(self):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
        Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
        """
        optimizer_D = self.optimizer[2]
        loss_D = self.loss[1]
        gen, discriminator = self.model
        total_loss = 0
        total_acc = 0
        total_sample_num = 0

        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            if random.random() < 0.85:
                continue
            batch_size = cond_data.shape[0]
            total_sample_num += batch_size
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            # real_gen_output_as_input = torch.cat([torch.zeros([batch_size, 1]), real_gen_output[:, :-1]], dim=1)
            fake, h_gen = gen.sample(cond_data)

            optimizer_D.zero_grad()
            # fake
            out, h_dis_fake = discriminator.batchClassify(cond_data, fake)
            loss = loss_D(out, torch.zeros(batch_size, device=cond_data.device))
            total_acc += torch.sum(out < 0.5).data.item()
            # real
            out, h_dis_real = discriminator.batchClassify(cond_data, real_gen_output)
            loss += loss_D(out, torch.ones(batch_size, device=cond_data.device))
            total_acc += torch.sum(out > 0.5).data.item()

            loss.backward()
            optimizer_D.step()

            total_loss += loss.data.item()

        self.logger.info(str(fake[0].cpu().detach().numpy().tolist()))
        self.logger.info(str(real_gen_output[0].cpu().detach().numpy().tolist()))

        avg_loss = total_loss / total_sample_num
        avg_acc = total_acc / total_sample_num / 2

        sys.stdout.flush()
        self.logger.info(' average_loss = %.4f, train_acc = %.4f' % (
            avg_loss, avg_acc))

        return avg_acc
