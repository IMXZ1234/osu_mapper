import os
import random
import sys

import torch
from torch import nn
from tqdm import tqdm

from system.base_sys import Train
from util.general_util import recursive_wrap_data
from util.plt_util import plot_loss


class TrainGAN(Train):
    def __init__(self, config_dict, train_type):
        super(TrainGAN, self).__init__(config_dict, train_type)

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
