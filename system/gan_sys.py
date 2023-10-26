import collections
import os
import random
import sys

import tensorboardX
import torch
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm
import numpy as np

from system.base_sys import Train
from util.general_util import recursive_wrap_data, dynamic_import
from util.plt_util import plot_loss, plot_signal
from util.train_util import MultiStepScheduler, idx_set_with_uniform_itv, AvgLossLogger, AbsCosineScheduler


class TrainGAN(Train):
    def __init__(self, config_dict, train_type, **kwargs):
        super(TrainGAN, self).__init__(config_dict, train_type, **kwargs)

        self.phases = self.config_dict['train_arg'].get('phases', ['pre_gen', 'pre_dis', 'adv', ])
        self.phase_epochs = self.config_dict['train_arg'].get('phase_epochs', [0 for _ in range(len(self.phases))])
        assert len(self.phases) == len(self.phase_epochs)
        self.start_phase = self.config_dict['train_arg'].get('start_phase', self.phases[0])
        self.start_epoch = self.config_dict['train_arg'].get('start_epoch', 0)
        print(self.phases, self.phase_epochs)
        print('start phase %s' % self.start_phase, 'start epoch %d' % self.start_epoch)

        self.save_train_state_itv = self.config_dict['train_arg'].get('save_train_state_itv', -1)

        if 'phase_train_func' in self.config_dict['train_arg']:
            func_names = self.config_dict['train_arg']['phase_train_func']
            self.phase_train_func = [
                getattr(self, fn)
                for fn in func_names
            ]
        else:
            self.phase_train_func = [
                self.train_discriminator,
                self.train_generator,
                self.run_adv_training,
            ]
        assert len(self.phases) == len(self.phase_train_func)

    def save_phase_train_state(self, phase, epoch):
        state = {'model': {},
                 'optimizer': {}}

        model = self.model
        if not isinstance(model, (list, tuple)):
            model = [self.model]
        for i, m in enumerate(model):
            state['model'][i] = m.state_dict()
        optimizer = self.optimizer
        if not isinstance(optimizer, (list, tuple)):
            optimizer = [self.optimizer]
        for i, opt in enumerate(optimizer):
            state['optimizer'][i] = opt.state_dict()

        pt_filename = '%s_%d.pt' % (phase, epoch)
        filename = os.path.join(self.train_state_dir, pt_filename)
        torch.save(state, filename)

    def load_phase_train_state(self, phase, epoch):
        pt_filename = '%s_%d.pt' % (phase, epoch)
        filename = os.path.join(self.train_state_dir, pt_filename)
        if not os.path.exists(filename):
            return False
        state = torch.load(filename)

        model = self.model
        if not isinstance(model, (list, tuple)):
            model = [self.model]
        for i, m in enumerate(model):
            m.load_state_dict(state['model'][i])
        optimizer = self.optimizer
        if not isinstance(optimizer, (list, tuple)):
            optimizer = [self.optimizer]
        for i, opt in enumerate(optimizer):
            opt.load_state_dict(state['optimizer'][i])

    def init_train_state(self):
        custom_init_weight = self.custom_init_weight
        if self.continue_training:
            load_success = self.load_phase_train_state(self.start_phase, self.start_epoch)
            if load_success:
                custom_init_weight = None

        model = self.model
        if not isinstance(model, (list, tuple)):
            model = [self.model]
        for i, m in enumerate(model):
            if custom_init_weight is not None:
                m.apply(custom_init_weight)
            if isinstance(self.output_device, int):
                m.cuda(self.output_device)

        # DataParallel
        # self.logger.info('try to use data parallel on %s' % str(self.data_parallel_devices))
        if self.data_parallel_devices is not None:
            self.logger.info('using data parallel on %s' % str(self.data_parallel_devices))
            if isinstance(self.model, (list, tuple)):
                self.model = [nn.DataParallel(
                    m,
                    device_ids=self.data_parallel_devices,
                    output_device=self.output_device
                ) for m in self.model]
            else:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.data_parallel_devices,
                    output_device=self.output_device
                )
        self.logger.info('initialized train state, to cuda device %s' % str(self.output_device))

    def save_model(self, epoch=-1, model_index=None):
        if not isinstance(self.model, (list, tuple)):
            models = [self.model]
        else:
            models = self.model
        if model_index is None:
            model_index = list(range(len(models)))
        pt_filename = 'model_%s_%d_%d.pt'
        for index in model_index:
            model = models[index]
            torch.save(model.state_dict(), os.path.join(self.model_save_dir, pt_filename % (self.current_phase, epoch, index)))
        self.logger.info('saved model of phase %s, epoch %d under %s' % (self.current_phase, epoch, self.model_save_dir))

    def run_train(self):
        self.init_train_state()
        self.dis_loss_list, self.gen_loss_list = [], []
        self.current_phase_idx = self.phases.index(self.start_phase)
        while self.current_phase_idx < len(self.phases):
            self.current_phase = self.phases[self.current_phase_idx]
            self.current_epoch = self.start_epoch
            self.current_phase_epochs = self.phase_epochs[self.current_phase_idx]
            self.current_phase_train_func = self.phase_train_func[self.current_phase_idx]

            print('starting phase %s, totally %d epochs' % (self.current_phase, self.current_phase_epochs))
            if self.start_epoch < self.current_phase_epochs:
                self.current_phase_train_func()
                self.save_phase_train_state(self.current_phase, self.current_epoch)
            self.start_epoch = 0
            self.current_phase_idx += 1

        self.save_model(-1)
        # log_dir = self.config_dict['output_arg']['log_dir']
        # plot_loss(self.dis_loss_list, 'discriminator loss',
        #           save_path=os.path.join(log_dir, 'discriminator_loss.png'), show=True)
        # plot_loss(self.gen_loss_list, 'generator loss',
        #           save_path=os.path.join(log_dir, 'generator_loss.png'), show=True)
        # self.save_properties()

    def run_adv_training(self):
        # # ADVERSARIAL TRAINING
        self.logger.info('\nStarting Adversarial Training...')
        acc_limit = self.config_dict['train_arg']['acc_limit'] if 'acc_limit' in self.config_dict['train_arg'] else 0.6

        for epoch in range(self.epoch):
            self.logger.info('\n--------\nEPOCH %d\n--------' % (epoch + 1))
            if self.config_dict['train_arg']['adaptive_adv_train']:
                while self.train_generator_epoch() > acc_limit:
                    pass
                while self.train_discriminator_epoch() < acc_limit:
                    pass
            else:
                # TRAIN GENERATOR
                self.logger.info('\nAdversarial Training Generator : ')
                for i in range(self.config_dict['train_arg']['adv_generator_epoch']):
                    self.train_generator_epoch()

                # TRAIN DISCRIMINATOR
                self.logger.info('\nAdversarial Training Discriminator : ')
                for i in range(self.config_dict['train_arg']['adv_discriminator_epoch']):
                    self.train_discriminator_epoch()
            if (epoch + 1) % self.model_save_step == 0:
                self.save_model(epoch, (0,))

    def on_epoch_end(self):
        if self.save_train_state_itv > 0 and (self.current_epoch+1) % self.save_train_state_itv == 0:
            self.save_phase_train_state(self.current_phase, self.current_epoch)

        if (self.current_epoch + 1) % self.model_save_step == 0:
            self.save_model(self.current_epoch, (0,))

    def train_generator_epoch(self):
        """
        The generator is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        optimizer_G, optimizer_D = self.optimizer[1], self.optimizer[2]
        # loss_G, loss_D = self.loss
        gen, dis = self.model
        epoch_dis_acc = 0
        epoch_pg_loss = 0
        epoch_gen_loss = 0
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

    def train_discriminator_epoch(self):
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

        self.logger.info(' average_loss = %.4f, train_acc = %.4f' % (
            avg_loss, avg_acc))

        sample = gen(cond_data)[0].cpu().detach().numpy()
        self.logger.info(sample[:])

        return avg_acc

    def train_generator(self):
        """
        The generator is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        for epoch in range(self.start_epoch, self.current_phase_epochs):
            print('epoch %d' % epoch)
            self.current_epoch = epoch
            epoch_dis_acc = self.train_generator_epoch()

            self.logger.info('epoch_dis_acc %.3f' % epoch_dis_acc)

            self.on_epoch_end()

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

        for epoch in range(self.start_epoch, self.current_phase_epochs):
            print('epoch %d' % epoch)
            self.current_epoch = epoch
            self.train_discriminator_epoch()

            self.logger.info(' average_loss = %.4f, train_acc = %.4f' % (
                avg_loss, avg_acc))

            self.on_epoch_end()


class TrainACGANWithinBatch(TrainGAN):
    def __init__(self, config_dict, train_type='classification', **kwargs):
        super(TrainACGANWithinBatch, self).__init__(config_dict, train_type, **kwargs)

        if 'embedding_decoder' in self.config_dict:
            self.embedding_output_decoder = dynamic_import(self.config_dict['embedding_decoder'])(**self.config_dict['embedding_decoder_args'])
        self.lambda_cls = self.config_dict['train_arg']['lambda_cls']

        log_dir = self.config_dict['output_arg']['log_dir']
        self.tensorboard_writer = tensorboardX.SummaryWriter(log_dir)

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
        noise_level_sched = AbsCosineScheduler(train_arg['noise_level_step'], train_arg['noise_level'], train_arg['period'])
        # noise_level_sched = MultiStepScheduler(train_arg['noise_level_step'], train_arg['noise_level'])

        adv_generator_epoch_sched.set_current_step(self.current_epoch-1)
        noise_level_sched.set_current_step(self.current_epoch-1)

        for epoch in range(self.current_epoch, self.epoch):
            time_cost_dict = {'net': 0, 'data': 0, 'log': 0}
            self.logger.info('\n--------\nEPOCH %d\n--------' % epoch)
            self.logger.info('lr %.8f' % self.optimizer[0].param_groups[0]['lr'])
            self.tensorboard_writer.add_scalar('lr', self.optimizer[0].param_groups[0]['lr'], epoch)
            epoch_gen_loss = 0
            total_loss = 0
            total_sample_num = 0

            epoch_fake_loss = 0
            epoch_real_loss = 0
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
            dis_pred = []

            self.log_time_stamp()
            for batch, (cond_data, real_gen_output, cls_label) in enumerate(tqdm(self.train_iter)):
                batch_size = cond_data.shape[0]
                total_sample_num += batch_size
                cond_data = recursive_wrap_data(cond_data, self.output_device)
                real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
                cls_label = recursive_wrap_data(cls_label, self.output_device)
                time_cost_dict['data'] += self.log_time_stamp()
                # print(cls_label.shape)

                # train generator batch
                if batch in train_gen_batches or (epoch_gen_loss / total_sample_num) > 5:
                    gen_loss = self.train_generator_batch((cond_data, real_gen_output, cls_label))
                    epoch_gen_loss += gen_loss * batch_size

                if len(self.exp_replay_buffer) > 0:
                    if self.exp_replay_buffer[0][-1] == 0:
                        exp_replay_cond_data, exp_replay_real_gen_output, exp_replay_fake, exp_replay_cls_label, _ = self.exp_replay_buffer.popleft()
                        loss, fake_loss, real_loss, cls_loss = self.train_discriminator_batch_exp_replay(
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
                    loss, fake_loss, real_loss, cls_loss, dis_pred_real, dis_pred_fake = self.train_discriminator_batch((cond_data, real_gen_output, cls_label))

                    dis_pred.append(dis_pred_real > 0.5)
                    dis_pred.append(dis_pred_fake <= 0.5)

                    # we only concern fake loss value
                    # if fake loss value is too large, generator training will be difficult
                    last_few_batch_fake_loss.append(fake_loss)
                    last_few_batch_real_loss.append(real_loss)

                    total_loss += loss * batch_size
                    epoch_fake_loss += fake_loss * batch_size
                    epoch_real_loss += real_loss * batch_size
                    epoch_cls_loss += cls_loss * batch_size
                time_cost_dict['net'] += self.log_time_stamp()

            gen_sched.step()
            dis_sched.step()
            adv_generator_epoch_sched.step()
            noise_level_sched.step()

            avg_gen_loss = epoch_gen_loss / total_sample_num
            self.logger.info('avg_gen_loss %.8f' % avg_gen_loss)
            self.tensorboard_writer.add_scalar('avg_gen_loss', avg_gen_loss, epoch)
            self.gen_loss_list.append(avg_gen_loss)

            print('generating sample...')
            print('run1')
            sample = gen((cond_data, cls_label))
            self.log_gen_output_embedding(sample, epoch, 'run1')
            print('run2')
            sample = gen((cond_data, cls_label))
            self.log_gen_output_embedding(sample, epoch, 'run2')
            print('max_difficulty')
            max_difficulty = torch.max(cls_label, dim=0, keepdim=True).values.expand_as(cls_label)
            sample = gen((cond_data, max_difficulty))
            self.log_gen_output_embedding(sample, epoch, str(max_difficulty[0].item()))
            print('min_difficulty')
            min_difficulty = torch.min(cls_label, dim=0, keepdim=True).values.expand_as(cls_label)
            sample = gen((cond_data, min_difficulty))
            self.log_gen_output_embedding(sample, epoch, str(min_difficulty[0].item()))

            avg_loss = total_loss / total_sample_num
            avg_fake_loss = epoch_fake_loss / total_sample_num
            avg_real_loss = epoch_real_loss / total_sample_num
            avg_cls_loss = epoch_cls_loss / total_sample_num
            self.dis_loss_list.append(avg_loss)
            dis_pred = torch.cat(dis_pred)
            self.logger.info('acc %.8f' % (torch.sum(dis_pred) / torch.numel(dis_pred)).item())

            self.logger.info('avg_loss = %.8f' % avg_loss)
            self.tensorboard_writer.add_scalar('avg_loss', avg_loss, epoch)
            self.logger.info('avg_fake_loss = %.8f' % avg_fake_loss)
            self.tensorboard_writer.add_scalar('avg_fake_loss', avg_fake_loss, epoch)
            self.logger.info('avg_real_loss = %.8f' % avg_real_loss)
            self.tensorboard_writer.add_scalar('avg_real_loss', avg_real_loss, epoch)
            self.logger.info('avg_cls_loss = %.8f' % avg_cls_loss)
            self.tensorboard_writer.add_scalar('avg_cls_loss', avg_cls_loss, epoch)

            if (epoch + 1) % self.model_save_step == 0:
                self.save_model(epoch, (0,))

            if (epoch + 1) % self.save_train_state_itv == 0:
                self.save_train_state(self.model[0], self.optimizer[0], epoch, 0)
                self.save_train_state(self.model[1], self.optimizer[1], epoch, 1)

            time_cost_dict['log'] += self.log_time_stamp()

            total_time = sum(time_cost_dict.values())
            for k, v in time_cost_dict.items():
                print('%s: %.2f' % (k, v / total_time), end=' ')

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
        real_loss = F.binary_cross_entropy(
            dis_real_validity_out,
            torch.ones(dis_real_validity_out.shape, device=dis_real_validity_out.device, dtype=torch.float32)
        )
        fake_loss = F.binary_cross_entropy(
            dis_fake_validity_out,
            torch.zeros(dis_fake_validity_out.shape, device=dis_fake_validity_out.device, dtype=torch.float32)
        )
        cls_loss = torch.nn.functional.mse_loss(
            torch.cat([dis_real_cls_out, dis_fake_cls_out], dim=0),
            torch.cat([cls_label, cls_label], dim=0)
        )
        # best both be negative
        loss = real_loss + fake_loss + self.lambda_cls * cls_loss

        loss.backward()

        if self.grad_alter_fn is not None:
            self.grad_alter_fn(dis.parameters())

        optimizer_D.step()

        return loss.item(), fake_loss.item(), real_loss.item(), cls_loss.item(), dis_real_validity_out, dis_fake_validity_out

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
        real_loss = F.binary_cross_entropy(
            dis_real_validity_out,
            torch.ones(dis_real_validity_out.shape, device=dis_real_validity_out.device, dtype=torch.float32)
        )
        fake_loss = F.binary_cross_entropy(
            dis_fake_validity_out,
            torch.zeros(dis_fake_validity_out.shape, device=dis_fake_validity_out.device, dtype=torch.float32)
        )
        cls_loss = torch.nn.functional.mse_loss(
            torch.cat([dis_real_cls_out, dis_fake_cls_out], dim=0),
            torch.cat([cls_label, cls_label], dim=0)
        )
        # best both be negative
        loss = real_loss + fake_loss + self.lambda_cls * cls_loss

        loss.backward()

        if self.grad_alter_fn is not None:
            self.grad_alter_fn(dis.parameters())

        optimizer_D.step()

        return loss.item(), fake_loss.item(), real_loss.item(), cls_loss.item()

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
        gen_loss = F.binary_cross_entropy(
            dis_validity_out,
            torch.ones(dis_validity_out.shape, device=dis_validity_out.device, dtype=torch.float32)
        )

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
        gen_output = gen_output[0].cpu().detach().numpy()
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

    def log_gen_output_embedding(self, gen_output, epoch, prefix='gen', plot_first=3):
        coord_output, embedding_output = gen_output
        coord_output = coord_output.cpu().detach().numpy()
        # print(embedding_output.shape)
        embedding_output = embedding_output.cpu().detach().numpy()
        # print(embedding_output.shape)
        output_dir = os.path.join(self.log_dir, 'output', str(epoch), prefix)
        os.makedirs(output_dir, exist_ok=True)
        all_sample_decoded = [
            self.embedding_output_decoder.decode(d)
            for d in embedding_output
        ]
        with open(os.path.join(output_dir, r'hit_signal.txt'), 'w') as f:
            for sample_decoded in all_sample_decoded:
                f.write(str(sample_decoded.tolist()))
        with open(os.path.join(output_dir, r'cursor_signal.txt'), 'w') as f:
            for sample_coord_output in coord_output:
                f.write(str(sample_coord_output.T.tolist()))
        for i, (sample_coord_output, sample_decoded) in enumerate(zip(coord_output, all_sample_decoded)):
            if i >= plot_first:
                break
            for signal, name in zip(
                sample_coord_output.T,
                [
                    'cursor_x', 'cursor_y'
                ]
            ):
                fig_array, fig = plot_signal(signal, name,
                                             save_path=os.path.join(output_dir, name + '%d.jpg' % i),
                                             show=False)
                self.tensorboard_writer.add_figure(prefix + name, fig, epoch)
            for signal, name in zip(
                [
                    np.where(sample_decoded == 1, 1, 0),
                    np.where(sample_decoded == 2, 1, 0),
                    np.where(sample_decoded == 3, 1, 0),
                ],
                [
                    'circle_hit', 'slider_hit', 'spinner_hit'
                ]
            ):
                fig_array, fig = plot_signal(signal, name,
                                             save_path=os.path.join(output_dir, name + '%d.jpg' % i),
                                             show=False)
                self.tensorboard_writer.add_figure(prefix + name, fig, epoch)
