import collections
import os
import random
import sys

import numpy as np
import torch
from torch import autograd
from tqdm import tqdm

from system.gan_sys import TrainGAN
from util.general_util import recursive_wrap_data
from util.plt_util import plot_loss, plot_signal
from util.train_util import MultiStepScheduler, idx_set_with_uniform_itv, AvgLossLogger
import tensorboardX


class TrainWGAN(TrainGAN):
    def __init__(self, config_dict, train_type='classification', **kwargs):
        super(TrainWGAN, self).__init__(config_dict, train_type, **kwargs)

        if 'lambda_gp' in self.config_dict['train_arg']:
            self.lambda_gp = self.config_dict['train_arg']['lambda_gp']
        else:
            self.lambda_gp = None
        # print('self.lambda_gp')
        # print(self.lambda_gp)
        log_dir = self.config_dict['output_arg']['log_dir']
        self.tensorboard_writer = tensorboardX.SummaryWriter(log_dir)

    def run_adv_training(self):
        # # ADVERSARIAL TRAINING
        self.logger.info('\nStarting Adversarial Training...')
        acc_limit = self.config_dict['train_arg']['acc_limit'] if 'acc_limit' in self.config_dict['train_arg'] else 0.6

        adv_generator_epoch, adv_discriminator_epoch =\
            self.config_dict['train_arg']['adv_generator_epoch'], self.config_dict['train_arg']['adv_discriminator_epoch']

        for epoch in range(self.epoch):
            self.logger.info('\n--------\nEPOCH %d\n--------' % (epoch + 1))
            # if self.config_dict['train_arg']['adaptive_adv_train']:
            for cur_gen_train_epoch in range(adv_generator_epoch):
                self.train_generator()
            for cur_dis_train_epoch in range(adv_discriminator_epoch):
                if self.train_discriminator() < -100:
                    break
            # else:
            #     # TRAIN GENERATOR
            #     self.logger.info('\nAdversarial Training Generator : ')
            #     for i in range(adv_generator_epoch):
            #         self.train_generator()
            #
            #     # TRAIN DISCRIMINATOR
            #     self.logger.info('\nAdversarial Training Discriminator : ')
            #     for i in range(adv_discriminator_epoch):
            #         self.train_discriminator()
            if (epoch + 1) % self.model_save_step == 0:
                self.save_model(epoch, (0,))

    def compute_gradient_penalty(self, cond_data, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        gen, dis = self.model
        batch_size = real_samples.shape[0]
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand([batch_size] + list(real_samples.shape[1:]), device=real_samples.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = dis(cond_data, interpolates)
        fake = torch.ones([batch_size, 1], dtype=torch.float, device=real_samples.device)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_generator(self):
        """
        Using Wesserstein GAN Loss
        """
        optimizer_G, optimizer_D = self.optimizer[0], self.optimizer[1]
        # loss_G, loss_D = self.loss
        gen, dis = self.model
        epoch_dis_acc = 0
        epoch_gen_loss = 0
        total_sample_num = 0
        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            batch_size = cond_data.shape[0]
            total_sample_num += batch_size
            cond_data = recursive_wrap_data(cond_data, self.output_device)

            fake = gen(cond_data)
            dis_cls_out = dis(cond_data, fake)
            # print('dis_cls_out')
            # print(dis_cls_out)
            epoch_dis_acc += torch.sum(dis_cls_out < 0.5).data.item()

            optimizer_G.zero_grad()

            # smaller the loss, better the generator performance
            gen_loss = -torch.mean(dis_cls_out)

            epoch_gen_loss += gen_loss.item() * batch_size

            gen_loss.backward()

            if self.grad_alter_fn is not None:
                self.grad_alter_fn(gen.parameters())

            optimizer_G.step()

        # epoch_dis_acc = epoch_dis_acc / total_sample_num
        # print('epoch_dis_acc %.3f' % epoch_dis_acc)
        epoch_gen_loss = epoch_gen_loss / total_sample_num
        self.logger.info('epoch_gen_loss %.8f' % epoch_gen_loss)
        self.gen_loss_list.append(epoch_gen_loss)

        sys.stdout.flush()
        # print('sample')
        sample = gen(cond_data)[0].cpu().detach().numpy()
        # self.logger.info(str(np.where(sample[:, 0] > 0.5, 1, 0)[:32]))
        # self.logger.info(str(np.where(sample[:, 1] > 0.5, 2, 0)[:32]))
        # self.logger.info(str(sample[:, 2:][:16]))

        return epoch_gen_loss

    def train_discriminator(self):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
        Samples are drawn d_steps times, and the discriminator is trained for 'epochs' epochs.
        """
        optimizer_D = self.optimizer[1]
        # loss_D = self.loss[1]
        gen, dis = self.model
        total_loss = 0
        total_sample_num = 0

        epoch_gp = 0
        epoch_fake_loss = 0
        epoch_real_loss = 0
        win_len = 10
        last_few_batch_loss = collections.deque([0 for _ in range(win_len)])
        win_avg_loss = 0

        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            # if random.random() < 0.85:
            #     continue
            batch_size = cond_data.shape[0]
            total_sample_num += batch_size
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            # real_gen_output_as_input = torch.cat([torch.zeros([batch_size, 1]), real_gen_output[:, :-1]], dim=1)
            optimizer_D.zero_grad()
            # real
            dis_real_cls_out = dis(cond_data, real_gen_output)
            with torch.no_grad():
                fake = gen(cond_data)

            # print(fake.requires_grad)
            # fake
            dis_fake_cls_out = dis(cond_data, fake)
            # optimization direction: larger score for real samples, smaller score for fake samples
            # smaller the loss, better the discriminator performance
            real_loss = -torch.mean(dis_real_cls_out)
            fake_loss = torch.mean(dis_fake_cls_out)
            # best both be negative
            loss = real_loss + fake_loss

            total_loss += loss.item() * batch_size
            epoch_fake_loss += fake_loss.item() * batch_size
            epoch_real_loss += real_loss.item() * batch_size

            if self.lambda_gp is not None:
                # use wgan gradient penalty
                gradient_penalty = self.compute_gradient_penalty(cond_data, real_gen_output.data, fake.data)
                gp_loss = self.lambda_gp * gradient_penalty
                loss = loss + gp_loss
                epoch_gp += gradient_penalty.item()

            # we only concern fake loss value
            # if fake loss value is too large, generator training will be difficult
            win_avg_loss = win_avg_loss + (fake_loss - last_few_batch_loss.popleft()) / win_len
            last_few_batch_loss.append(fake_loss)
            # if win_avg_loss < -100:
            #     break

            loss.backward()

            # if self.grad_alter_fn is not None:
            #     self.grad_alter_fn(dis.parameters())

            optimizer_D.step()

            # Clip weights of discriminator
            if self.lambda_gp is None:
                clip_value = 1
                for p in dis.parameters():
                    p.data.clamp_(-clip_value, clip_value)

        avg_loss = total_loss / total_sample_num
        avg_fake_loss = epoch_fake_loss / total_sample_num
        avg_real_loss = epoch_real_loss / total_sample_num
        self.dis_loss_list.append(avg_loss)

        sys.stdout.flush()
        self.logger.info('average_loss = %.8f' % avg_loss)
        self.logger.info('average_fake_loss = %.8f' % avg_fake_loss)
        self.logger.info('average_real_loss = %.8f' % avg_real_loss)
        if self.lambda_gp is not None:
            self.logger.info('gp = %.8f' % (epoch_gp / len(self.train_iter)))

        return avg_loss


class TrainWGANWithinBatch(TrainWGAN):
    def __init__(self, config_dict, train_type='classification', **kwargs):
        super(TrainWGANWithinBatch, self).__init__(config_dict, train_type, **kwargs)

        if 'lambda_gp' in self.config_dict['train_arg']:
            self.lambda_gp = self.config_dict['train_arg']['lambda_gp']
        else:
            self.lambda_gp = None
        # print('self.lambda_gp')
        # print(self.lambda_gp)

    def run_train(self):
        self.epoch_gp_loss = []

        super(TrainWGANWithinBatch, self).run_train()

        log_dir = self.config_dict['output_arg']['log_dir']
        plot_loss(self.gen_loss_list, 'generator loss',
                  save_path=os.path.join(log_dir, 'generator_loss.png'), show=True)

    def run_adv_training(self):
        # # ADVERSARIAL TRAINING
        self.logger.info('\nStarting Adversarial Training...')

        train_arg = self.config_dict['train_arg']
        init_adv_generator_epoch, init_adv_discriminator_epoch =\
            train_arg['adv_generator_epoch'], train_arg['adv_discriminator_epoch']
        # decrease generator's probability of getting trained
        gen_lambda = train_arg['gen_lambda']
        gen_lambda_step = train_arg['gen_lambda_step']

        gen, dis = self.model
        gen_sched, dis_sched = self.scheduler

        adv_generator_epoch_sched = MultiStepScheduler(train_arg['gen_lambda_step'], train_arg['gen_lambda'])

        for epoch in range(self.epoch):
            self.logger.info('\n--------\nEPOCH %d\n--------' % (epoch + 1))
            self.logger.info('lr %.8f' % self.optimizer[0].param_groups[0]['lr'])
            epoch_gen_loss = 0
            total_loss = 0
            total_sample_num = 0

            epoch_fake_loss = 0
            epoch_real_loss = 0
            epoch_gp_loss = 0
            win_len = 10
            last_few_batch_fake_loss = collections.deque([0 for _ in range(win_len)])
            last_few_batch_real_loss = collections.deque([0 for _ in range(win_len)])
            win_avg_fake_loss = 0
            win_avg_real_loss = 0

            adv_generator_epoch = init_adv_generator_epoch * adv_generator_epoch_sched.cur_milestone_output()
            self.logger.info('adv_generator_epoch %.8f' % adv_generator_epoch)

            epoch_num_batches = len(self.train_iter)
            train_gen_num_batches = round(adv_generator_epoch * epoch_num_batches)
            train_dis_num_batches = round(init_adv_discriminator_epoch * epoch_num_batches)
            train_gen_batches = idx_set_with_uniform_itv(epoch_num_batches, train_gen_num_batches)
            train_dis_batches = idx_set_with_uniform_itv(epoch_num_batches, train_dis_num_batches)

            for batch_idx, all_batch_items in enumerate(tqdm(self.train_iter)):
                batch, other = all_batch_items[:-1], all_batch_items[-1]
                batch_size = other.shape[0]
                total_sample_num += batch_size
                batch = recursive_wrap_data(batch, self.output_device)

                # train generator batch
                if batch_idx in train_gen_batches or (epoch_gen_loss / total_sample_num) > 5:
                    gen_loss = self.train_generator_batch(batch)
                    epoch_gen_loss += gen_loss * batch_size

                # train discriminator batch
                if batch_idx in train_dis_batches:
                    # if abs(win_avg_fake_loss) > 100 or abs(win_avg_real_loss) > 100 or (epoch_gen_loss / total_sample_num) > 10:
                    #     continue
                    loss, fake_loss, real_loss, gp_loss = self.train_discriminator_batch(batch)

                    # we only concern fake loss value
                    # if fake loss value is too large, generator training will be difficult
                    win_avg_fake_loss = win_avg_fake_loss + (fake_loss - last_few_batch_fake_loss.popleft()) / win_len
                    last_few_batch_fake_loss.append(fake_loss)
                    win_avg_real_loss = win_avg_real_loss + (real_loss - last_few_batch_real_loss.popleft()) / win_len
                    last_few_batch_real_loss.append(real_loss)

                    total_loss += loss * batch_size
                    epoch_fake_loss += fake_loss * batch_size
                    epoch_real_loss += real_loss * batch_size
                    epoch_gp_loss += gp_loss * batch_size

            gen_sched.step()
            dis_sched.step()
            adv_generator_epoch_sched.step()

            avg_gen_loss = epoch_gen_loss / total_sample_num
            self.logger.info('avg_gen_loss %.8f' % avg_gen_loss)
            self.gen_loss_list.append(avg_gen_loss)

            # sample = gen(cond_data)[0].cpu().detach().numpy()
            # self.logger.info(str(np.where(sample[:, 0] > 0.5, 1, 0)[:32]))
            # self.logger.info(str(np.where(sample[:, 1] > 0.5, 2, 0)[:32]))
            # self.logger.info('\n' + str(sample[:, 2:][:16]))

            avg_loss = total_loss / total_sample_num
            avg_fake_loss = epoch_fake_loss / total_sample_num
            avg_real_loss = epoch_real_loss / total_sample_num
            self.dis_loss_list.append(avg_loss)

            self.logger.info('avg_loss = %.8f' % avg_loss)
            self.logger.info('avg_fake_loss = %.8f' % avg_fake_loss)
            self.logger.info('avg_real_loss = %.8f' % avg_real_loss)
            if self.lambda_gp is not None:
                self.logger.info('avg_gp_loss = %.8f' % (epoch_gp_loss / len(self.train_iter)))

    def compute_gradient_penalty(self, cond_data, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        gen, dis = self.model
        batch_size = real_samples.shape[0]
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand([batch_size] + list(real_samples.shape[1:]), device=real_samples.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates_validity = dis(cond_data, interpolates)
        fake = torch.ones([batch_size, 1], dtype=torch.float, device=real_samples.device, requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=interpolates_validity,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_generator_batch(self, batch_items):
        """
        Using Wesserstein GAN Loss
        """
        cond_data, = batch_items

        optimizer_G, optimizer_D = self.optimizer[0], self.optimizer[1]
        # loss_G, loss_D = self.loss
        gen, dis = self.model

        sys.stdout.flush()

        fake = gen(cond_data)
        dis_cls_out = dis(cond_data, fake)

        optimizer_G.zero_grad()

        # smaller the loss, better the generator performance
        gen_loss = -torch.mean(dis_cls_out)

        gen_loss.backward()

        if self.grad_alter_fn is not None:
            self.grad_alter_fn(gen.parameters())

        optimizer_G.step()

        return gen_loss.item()

    def train_discriminator_batch(self, batch_items):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
        Samples are drawn d_steps times, and the discriminator is trained for 'epochs' epochs.
        """
        cond_data, real_gen_output = batch_items

        optimizer_D = self.optimizer[1]
        # loss_D = self.loss[1]
        gen, dis = self.model

        # real_gen_output_as_input = torch.cat([torch.zeros([batch_size, 1]), real_gen_output[:, :-1]], dim=1)
        optimizer_D.zero_grad()
        # real
        dis_real_cls_out = dis(cond_data, real_gen_output)
        with torch.no_grad():
            fake = gen(cond_data)
            if random.random() < self.log_exp_replay_prob:
                self.exp_replay_buffer.append([cond_data, fake, real_gen_output, self.exp_replay_wait])

        # print('fake')
        dis_fake_cls_out = dis(cond_data, fake)
        # optimization direction: larger score for real samples, smaller score for fake samples
        # smaller the loss, better the discriminator performance
        real_loss = -torch.mean(dis_real_cls_out)
        fake_loss = torch.mean(dis_fake_cls_out)
        # best both be negative
        loss = real_loss + fake_loss

        if self.lambda_gp is not None:
            # use wgan gradient penalty
            gradient_penalty = self.compute_gradient_penalty(cond_data, real_gen_output.data, fake.data)
            gp_loss = self.lambda_gp * gradient_penalty
            loss = loss + gp_loss
        else:
            gp_loss = 0

        loss.backward()

        # if self.grad_alter_fn is not None:
        #     self.grad_alter_fn(dis.parameters())

        optimizer_D.step()

        # Clip weights of discriminator
        if self.lambda_gp is None:
            clip_value = 1
            for p in dis.parameters():
                p.data.clamp_(-clip_value, clip_value)

        return loss.item(), fake_loss.item(), real_loss.item(), gp_loss.item()

    def train_discriminator_batch_exp_replay(self, batch_items):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
        Samples are drawn d_steps times, and the discriminator is trained for 'epochs' epochs.
        """
        cond_data, real_gen_output, fake = batch_items

        optimizer_D = self.optimizer[1]
        # loss_D = self.loss[1]
        gen, dis = self.model

        # real_gen_output_as_input = torch.cat([torch.zeros([batch_size, 1]), real_gen_output[:, :-1]], dim=1)
        optimizer_D.zero_grad()
        # real
        dis_real_cls_out = dis(cond_data, real_gen_output)
        dis_fake_cls_out = dis(cond_data, fake)

        # optimization direction: larger score for real samples, smaller score for fake samples
        # smaller the loss, better the discriminator performance
        real_loss = -torch.mean(dis_real_cls_out)
        fake_loss = torch.mean(dis_fake_cls_out)
        # best both be negative
        loss = real_loss + fake_loss

        if self.lambda_gp is not None:
            # use wgan gradient penalty
            gradient_penalty = self.compute_gradient_penalty(cond_data, real_gen_output.data, fake.data)
            gp_loss = self.lambda_gp * gradient_penalty
            loss = loss + gp_loss
        else:
            gp_loss = 0

        loss.backward()

        # if self.grad_alter_fn is not None:
        #     self.grad_alter_fn(dis.parameters())

        optimizer_D.step()

        # Clip weights of discriminator
        if self.lambda_gp is None:
            clip_value = 1
            for p in dis.parameters():
                p.data.clamp_(-clip_value, clip_value)

        return loss.item(), fake_loss.item(), real_loss.item(), gp_loss.item()


class TrainACWGANWithinBatch(TrainWGANWithinBatch):
    def __init__(self, config_dict, train_type='classification', **kwargs):
        super(TrainACWGANWithinBatch, self).__init__(config_dict, train_type, **kwargs)

        if 'lambda_gp' in self.config_dict['train_arg']:
            self.lambda_gp = self.config_dict['train_arg']['lambda_gp']
        else:
            self.lambda_gp = None
        self.lambda_cls = self.config_dict['train_arg']['lambda_cls']

    def run_train(self):
        print('Start Train...')

        super(TrainWGANWithinBatch, self).run_train()

        # plot_loss(self.gen_loss_list, 'generator loss',
        #           save_path=os.path.join(self.log_dir, 'generator_loss.png'), show=True)

    def run_adv_training(self):
        # # ADVERSARIAL TRAINING
        self.logger.info('\nStarting Adversarial Training...')

        train_arg = self.config_dict['train_arg']
        init_adv_generator_epoch, init_adv_discriminator_epoch =\
            train_arg['adv_generator_epoch'], train_arg['adv_discriminator_epoch']
        # # decrease generator's probability of getting trained
        # gen_lambda = train_arg['gen_lambda']
        # gen_lambda_step = train_arg['gen_lambda_step']
        self.log_exp_replay_prob = train_arg['log_exp_replay_prob']

        self.exp_replay_buffer = collections.deque()
        self.exp_replay_wait = train_arg['exp_replay_wait']

        gen, dis = self.model
        gen_sched, dis_sched = self.scheduler

        adv_generator_epoch_sched = MultiStepScheduler(train_arg['gen_lambda_step'], train_arg['gen_lambda'])
        noise_level_sched = MultiStepScheduler(train_arg['noise_level_step'], train_arg['noise_level'])

        adv_generator_epoch_sched.set_current_step(self.current_epoch-1)
        noise_level_sched.set_current_step(self.current_epoch-1)

        for epoch in range(self.current_epoch, self.epoch):
            self.logger.info('\n--------\nEPOCH %d\n--------' % epoch)
            self.logger.info('lr %.8f' % self.optimizer[0].param_groups[0]['lr'])
            self.tensorboard_writer.add_scalar('lr', self.optimizer[0].param_groups[0]['lr'], epoch)
            epoch_gen_loss = 0
            total_loss = 0
            total_sample_num = 0

            epoch_fake_loss = 0
            epoch_real_loss = 0
            epoch_gp_loss = 0
            epoch_cls_loss = 0

            adv_generator_epoch = init_adv_generator_epoch * adv_generator_epoch_sched.cur_milestone_output()
            self.logger.info('adv_generator_epoch %.8f' % adv_generator_epoch)
            self.tensorboard_writer.add_scalar('adv_generator_epoch', adv_generator_epoch, epoch)
            noise_level = noise_level_sched.cur_milestone_output()
            self.logger.info('noise_level %.8f' % noise_level)
            self.tensorboard_writer.add_scalar('noise_level', noise_level, epoch)

            self.train_iter.dataset.set_noise_level(noise_level)

            epoch_num_batches = len(self.train_iter)
            train_gen_num_batches = round(adv_generator_epoch * epoch_num_batches)
            train_dis_num_batches = round(init_adv_discriminator_epoch * epoch_num_batches)
            train_gen_batches = idx_set_with_uniform_itv(epoch_num_batches, train_gen_num_batches)
            train_dis_batches = idx_set_with_uniform_itv(epoch_num_batches, train_dis_num_batches)

            last_few_batch_fake_loss = AvgLossLogger(10)
            last_few_batch_real_loss = AvgLossLogger(10)

            for batch, (cond_data, real_gen_output, cls_label) in enumerate(tqdm(self.train_iter)):
                batch_size = cond_data.shape[0]
                total_sample_num += batch_size
                cond_data = recursive_wrap_data(cond_data, self.output_device)
                real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
                cls_label = recursive_wrap_data(cls_label, self.output_device)
                # print(cls_label.shape)

                # train generator batch
                if batch in train_gen_batches or (epoch_gen_loss / total_sample_num) > 5:
                    gen_loss = self.train_generator_batch((cond_data, real_gen_output, cls_label))
                    epoch_gen_loss += gen_loss * batch_size

                if len(self.exp_replay_buffer) > 0:
                    if self.exp_replay_buffer[0][-1] == 0:
                        exp_replay_cond_data, exp_replay_real_gen_output, exp_replay_fake, exp_replay_cls_label, _ = self.exp_replay_buffer.popleft()
                        loss, fake_loss, real_loss, gp_loss = self.train_discriminator_batch_exp_replay(
                            (exp_replay_cond_data, exp_replay_real_gen_output, exp_replay_fake, exp_replay_cls_label)
                        )
                        # self.logger.info('exp replay batch: ')
                        # self.logger.info('loss = %.8f' % loss)
                        # self.logger.info('fake_loss = %.8f' % fake_loss)
                        # self.logger.info('real_loss = %.8f' % real_loss)
                        # self.logger.info('gp_loss = %.8f' % gp_loss)
                        self.exp_replay_cur_wait = 0
                    for exp_replay_item in self.exp_replay_buffer:
                        exp_replay_item[-1] -= 1

                # train discriminator batch
                if batch in train_dis_batches:
                    # if abs(win_avg_fake_loss) > 100 or abs(win_avg_real_loss) > 100 or (epoch_gen_loss / total_sample_num) > 10:
                    #     continue
                    loss, fake_loss, real_loss, gp_loss, cls_loss = self.train_discriminator_batch((cond_data, real_gen_output, cls_label))

                    # we only concern fake loss value
                    # if fake loss value is too large, generator training will be difficult
                    last_few_batch_fake_loss.append(fake_loss)
                    last_few_batch_real_loss.append(real_loss)

                    total_loss += loss * batch_size
                    epoch_fake_loss += fake_loss * batch_size
                    epoch_real_loss += real_loss * batch_size
                    epoch_gp_loss += gp_loss * batch_size
                    epoch_cls_loss += cls_loss * batch_size

            gen_sched.step()
            dis_sched.step()
            adv_generator_epoch_sched.step()

            avg_gen_loss = epoch_gen_loss / total_sample_num
            self.logger.info('avg_gen_loss %.8f' % avg_gen_loss)
            self.tensorboard_writer.add_scalar('avg_gen_loss', avg_gen_loss, epoch)
            self.gen_loss_list.append(avg_gen_loss)

            sample = gen((cond_data, cls_label))[0].cpu().detach().numpy()
            self.log_gen_output(sample, epoch)

            avg_loss = total_loss / total_sample_num
            avg_fake_loss = epoch_fake_loss / total_sample_num
            avg_real_loss = epoch_real_loss / total_sample_num
            avg_cls_loss = epoch_cls_loss / total_sample_num
            self.dis_loss_list.append(avg_loss)

            self.logger.info('avg_loss = %.8f' % avg_loss)
            self.tensorboard_writer.add_scalar('avg_loss', avg_loss, epoch)
            self.logger.info('avg_fake_loss = %.8f' % avg_fake_loss)
            self.tensorboard_writer.add_scalar('avg_fake_loss', avg_fake_loss, epoch)
            self.logger.info('avg_real_loss = %.8f' % avg_real_loss)
            self.tensorboard_writer.add_scalar('avg_real_loss', avg_real_loss, epoch)
            self.logger.info('avg_cls_loss = %.8f' % avg_cls_loss)
            self.tensorboard_writer.add_scalar('avg_cls_loss', avg_cls_loss, epoch)
            if self.lambda_gp is not None:
                avg_gp_loss = epoch_gp_loss / total_sample_num
                self.logger.info('avg_gp_loss = %.8f' % avg_gp_loss)
                self.tensorboard_writer.add_scalar('avg_gp_loss', avg_gp_loss, epoch)

            if (epoch + 1) % self.model_save_step == 0:
                self.save_model(epoch, (0,))

            if (epoch + 1) % self.save_train_state_itv == 0:
                self.save_train_state(self.model[0], self.optimizer[0], epoch, 0)
                self.save_train_state(self.model[1], self.optimizer[1], epoch, 1)

    def train_discriminator_batch(self, batch_items):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
        Samples are drawn d_steps times, and the discriminator is trained for 'epochs' epochs.
        """
        cond_data, real_gen_output, cls_label = batch_items

        optimizer_D = self.optimizer[1]
        # loss_D = self.loss[1]
        gen, dis = self.model

        # real_gen_output_as_input = torch.cat([torch.zeros([batch_size, 1]), real_gen_output[:, :-1]], dim=1)
        optimizer_D.zero_grad()
        # real
        dis_real_validity_out, dis_real_cls_out = dis(cond_data, real_gen_output)
        with torch.no_grad():
            fake = gen((cond_data, cls_label))
            if random.random() < self.log_exp_replay_prob:
                self.exp_replay_buffer.append([cond_data, real_gen_output, fake, cls_label, self.exp_replay_wait])

        # print('fake')
        dis_fake_validity_out, dis_fake_cls_out = dis(cond_data, fake)
        # optimization direction: larger score for real samples, smaller score for fake samples
        # smaller the loss, better the discriminator performance
        real_loss = -torch.mean(dis_real_validity_out)
        fake_loss = torch.mean(dis_fake_validity_out)
        cls_loss = torch.nn.functional.mse_loss(
            torch.cat([dis_real_cls_out, dis_fake_cls_out], dim=0),
            torch.cat([cls_label, cls_label], dim=0)
        )
        # best both be negative
        loss = real_loss + fake_loss + self.lambda_cls * cls_loss

        if self.lambda_gp is not None:
            # use wgan gradient penalty
            gradient_penalty = self.compute_gradient_penalty(cond_data, real_gen_output.data, fake.data)
            gp_loss = self.lambda_gp * gradient_penalty
            loss = loss + gp_loss
        else:
            gp_loss = 0

        loss.backward()

        # if self.grad_alter_fn is not None:
        #     self.grad_alter_fn(dis.parameters())

        optimizer_D.step()

        # Clip weights of discriminator
        if self.lambda_gp is None:
            clip_value = 1
            for p in dis.parameters():
                p.data.clamp_(-clip_value, clip_value)

        return loss.item(), fake_loss.item(), real_loss.item(), gp_loss.item(), cls_loss.item()

    def train_discriminator_batch_exp_replay(self, batch_items):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
        Samples are drawn d_steps times, and the discriminator is trained for 'epochs' epochs.
        """
        cond_data, real_gen_output, fake, cls_label = batch_items

        optimizer_D = self.optimizer[1]
        # loss_D = self.loss[1]
        gen, dis = self.model

        # real_gen_output_as_input = torch.cat([torch.zeros([batch_size, 1]), real_gen_output[:, :-1]], dim=1)
        optimizer_D.zero_grad()
        # real
        dis_real_validity_out, dis_real_cls_out = dis(cond_data, real_gen_output)
        dis_fake_validity_out, dis_fake_cls_out = dis(cond_data, fake)

        # optimization direction: larger score for real samples, smaller score for fake samples
        # smaller the loss, better the discriminator performance
        real_loss = -torch.mean(dis_real_validity_out)
        fake_loss = torch.mean(dis_fake_validity_out)
        cls_loss = torch.nn.functional.mse_loss(
            torch.cat([dis_real_cls_out, dis_fake_cls_out], dim=0),
            torch.cat([cls_label, cls_label], dim=0)
        )
        # best both be negative
        loss = real_loss + fake_loss + self.lambda_cls * cls_loss

        if self.lambda_gp is not None:
            # use wgan gradient penalty
            gradient_penalty = self.compute_gradient_penalty(cond_data, real_gen_output.data, fake.data)
            gp_loss = self.lambda_gp * gradient_penalty
            loss = loss + gp_loss
        else:
            gp_loss = 0

        loss.backward()

        # if self.grad_alter_fn is not None:
        #     self.grad_alter_fn(dis.parameters())

        optimizer_D.step()

        # Clip weights of discriminator
        if self.lambda_gp is None:
            clip_value = 1
            for p in dis.parameters():
                p.data.clamp_(-clip_value, clip_value)

        return loss.item(), fake_loss.item(), real_loss.item(), gp_loss.item()

    def train_generator_batch(self, batch_items):
        """
        Using Wesserstein GAN Loss
        """
        cond_data, real_gen_output, cls_label = batch_items

        optimizer_G, optimizer_D = self.optimizer[0], self.optimizer[1]
        # loss_G, loss_D = self.loss
        gen, dis = self.model

        sys.stdout.flush()

        fake = gen((cond_data, cls_label))
        dis_validity_out, dis_cls_out = dis(cond_data, fake)

        optimizer_G.zero_grad()

        # smaller the loss, better the generator performance
        gen_loss = -torch.mean(dis_validity_out)

        cls_loss = torch.nn.functional.mse_loss(
            dis_cls_out,
            cls_label,
        )
        # best both be negative
        gen_loss = gen_loss + self.lambda_cls * cls_loss
        gen_loss.backward()

        if self.grad_alter_fn is not None:
            self.grad_alter_fn(gen.parameters())

        optimizer_G.step()

        return gen_loss.item()

    def log_gen_output(self, gen_output, epoch):
        output_dir = os.path.join(self.log_dir, 'output', str(epoch))
        os.makedirs(output_dir, exist_ok=True)
        for signal, name in zip(
            gen_output.T,
            [
                'circle_hit', 'slider_hit', 'spinner_hit', 'cursor_x', 'cursor_y'
            ]
        ):
            fig_array, fig = plot_signal(signal, name,
                                         save_path=os.path.join(output_dir, name + '.jpg'),
                                         show=False)
            self.tensorboard_writer.add_figure(name, fig, epoch)

    def compute_gradient_penalty(self, cond_data, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        gen, dis = self.model
        batch_size = real_samples.shape[0]
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand([batch_size] + list(real_samples.shape[1:]), device=real_samples.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates_validity, _ = dis(cond_data, interpolates)
        fake = torch.ones([batch_size, 1], dtype=torch.float, device=real_samples.device, requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=interpolates_validity,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
