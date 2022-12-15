import collections
import copy
import functools
import logging
import math
import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn, autograd
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm

from nn.dataset import collate_fn
from nn.metrics import default_metrices
from util.general_util import dynamic_import, recursive_to_cpu, recursive_wrap_data, try_format_dict_with_path
from util.plt_util import plot_loss
from util.train_util import MultiStepScheduler, idx_set_with_uniform_itv

np.set_printoptions(suppress=True)


class Train:
    def __init__(self,
                 config_dict,
                 task_type='classification',
                 **kwargs):
        self.task_type = task_type
        self.config_dict = config_dict
        model_arg, optimizer_arg, scheduler_arg, data_arg, loss_arg, pred_arg, output_arg, train_arg = \
            config_dict['model_arg'], config_dict['optimizer_arg'], config_dict['scheduler_arg'], config_dict['data_arg'], \
            config_dict['loss_arg'], config_dict['pred_arg'], config_dict['output_arg'], config_dict['train_arg']

        if 'output_device' in config_dict and config_dict['output_device'] is not None:
            self.output_device = config_dict['output_device']
        else:
            self.output_device = 'cpu'

        if 'train_type' in config_dict:
            self.train_type = config_dict['train_type']
            if self.train_type == 'gan':
                self.use_ext_cond_data = train_arg['use_ext_cond_data'] if 'use_ext_cond_data' in train_arg else False
                print(self.use_ext_cond_data)
            # self.use_random_iter = kwargs['use_random_iter']
        else:
            self.train_type = 'default'

        if self.task_type == 'classification':
            if 'num_classes' in config_dict and config_dict['num_classes'] is not None:
                self.num_classes = config_dict['num_classes']
            if 'cal_acc_func' in config_dict and config_dict['cal_acc_func'] is not None:
                self.cal_epoch_acc = dynamic_import(config_dict['cal_acc_func'])
            else:
                self.cal_epoch_acc = default_metrices.cal_acc_func
            if 'cal_cm_func' in config_dict and config_dict['cal_cm_func'] is not None:
                self.cal_epoch_cm = dynamic_import(config_dict['cal_cm_func'])
            else:
                self.cal_epoch_cm = default_metrices.cal_cm_func
            self.pred = self.load_pred(**pred_arg)
            self.train_pred_list = []
            self.test_pred_list = []
            self.train_accuracy_list = []
            self.eval_accuracy_list = []

        if 'collate_fn' in config_dict and config_dict['collate_fn'] is not None:
            self.collate_fn = dynamic_import(config_dict['collate_fn'])
        else:
            self.collate_fn = None
        if 'output_collate_fn' in config_dict and config_dict['output_collate_fn'] is not None:
            self.output_collate_fn = dynamic_import(config_dict['output_collate_fn'])
        else:
            self.output_collate_fn = collate_fn.output_collate_fn
        if 'grad_alter_fn' in config_dict and config_dict['grad_alter_fn'] is not None:
            if config_dict['grad_alter_fn'] == 'value':
                self.grad_alter_fn = functools.partial(nn.utils.clip_grad_value_, **config_dict['grad_alter_fn_arg'])
            elif config_dict['grad_alter_fn'] == 'norm':
                self.grad_alter_fn = functools.partial(nn.utils.clip_grad_norm_, **config_dict['grad_alter_fn_arg'])
            else:
                self.grad_alter_fn = dynamic_import(config_dict['grad_alter_fn'])
                self.grad_alter_fn_arg = config_dict['grad_alter_fn_arg']
        else:
            self.grad_alter_fn = None
        if 'train_extra' in config_dict and config_dict['train_extra'] is not None:
            self.train_extra = config_dict['train_extra']
        else:
            self.train_extra = dict()
        if 'test_extra' in config_dict and config_dict['test_extra'] is not None:
            self.test_extra = config_dict['test_extra']
        else:
            self.test_extra = dict()

        self.train_state_dir = config_dict['train_state_dir'] if 'train_state_dir' in config_dict else None

        self.start_epoch = 0
        self.writer, self.logger, self.log_dir, self.model_save_dir, self.model_save_step = self.load_logger(
            **output_arg)
        self.model = self.load_model(**model_arg)
        self.optimizer = self.load_optimizer(**optimizer_arg)
        self.scheduler = self.load_scheduler(**scheduler_arg)
        self.train_iter, self.test_iter = self.load_data(**data_arg)
        self.loss = self.load_loss(**loss_arg)

        self.epoch = train_arg['epoch']
        self.eval_step = train_arg['eval_step']

        self.control_file_path = os.path.join(self.log_dir, 'control.yaml')
        self.train_loss_list = []
        self.eva_loss_list = []
        self.learning_rate_list = []
        self.save_model_flag = False

    def load_logger(self, log_dir, model_save_dir, model_save_step, **kwargs):
        writer = SummaryWriter(logdir=log_dir + '/run')
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                      datefmt="%a %b %d %H:%M:%S %Y")

        sHandler = logging.StreamHandler()
        sHandler.setFormatter(formatter)
        logger.addHandler(sHandler)

        fHandler = logging.FileHandler(log_dir + '/log.txt', mode='w')
        fHandler.setLevel(logging.DEBUG)
        fHandler.setFormatter(formatter)
        logger.addHandler(fHandler)
        return writer, logger, log_dir, model_save_dir, model_save_step

    def load_model(self, model_type, **kwargs):
        if isinstance(model_type, (list, tuple)):
            net = [
                self.load_model(mt, **kwargs['params'][i])
                for i, mt in enumerate(model_type)
            ]
        else:
            net = dynamic_import(model_type)(**kwargs)
        self.logger.info('loaded model: %s' % model_type)
        return net

    def load_optimizer(self, optimizer_type, model_index=None, **kwargs):
        if isinstance(optimizer_type, (list, tuple)):
            if 'models_index' in kwargs:
                models_index = kwargs['models_index']
            else:
                assert len(optimizer_type) == len(self.model)
                models_index = list(range(len(self.model)))
            optimizer = [
                self.load_optimizer(ot, i, **kwargs['params'][i])
                for i, ot in zip(models_index, optimizer_type)
            ]
        else:
            if model_index is None:
                model = self.model
            else:
                model = self.model[model_index]
            if optimizer_type == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), **kwargs)
            elif optimizer_type == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), **kwargs)
            elif optimizer_type == 'RMSprop':
                optimizer = torch.optim.RMSprop(model.parameters(), **kwargs)
            else:
                optimizer = dynamic_import(optimizer_type)(model.parameters(), **kwargs)
        return optimizer

    def load_scheduler(self, scheduler_type, optimizer_index=None, **kwargs):
        if isinstance(scheduler_type, (list, tuple)):
            scheduler = [
                self.load_scheduler(st, i, **kwargs['params'][i])
                for i, st in enumerate(scheduler_type)
            ]
        else:
            if optimizer_index is None:
                optimizer = self.optimizer
            else:
                optimizer = self.optimizer[optimizer_index]
            if scheduler_type == 'StepLR':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
            elif scheduler_type == 'MultiStepLR':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
            else:
                scheduler = dynamic_import(scheduler_type)(optimizer, **kwargs)
        return scheduler

    def load_data(self,
                  dataset,
                  train_dataset_arg,
                  test_dataset_arg=None,
                  batch_size=8,
                  shuffle=True,
                  num_workers=1,
                  drop_last=False,
                  **kwargs):
        train_iter = torch.utils.data.DataLoader(
            dataset=dynamic_import(dataset)(**train_dataset_arg),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            drop_last=drop_last)
        if test_dataset_arg is not None:
            test_iter = torch.utils.data.DataLoader(
                dataset=dynamic_import(dataset)(**test_dataset_arg),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=num_workers,
                drop_last=False)
        else:
            test_iter = None
        return train_iter, test_iter

    def load_loss(self, loss_type, **kwargs):
        if isinstance(loss_type, (list, tuple)):
            loss = [
                self.load_loss(lt, **kwargs['params'][i])
                for i, lt in enumerate(loss_type)
            ]
        else:
            if loss_type == 'MSELoss':
                loss = nn.MSELoss(**kwargs)
            elif loss_type == 'BCELoss':
                loss = nn.BCELoss(**kwargs)
            elif loss_type == 'NLLLoss':
                loss = nn.NLLLoss(**kwargs)
            elif loss_type == 'CrossEntropyLoss':
                if 'weight' in kwargs and kwargs['weight'] is not None:
                    kwargs['weight'] = torch.tensor(kwargs['weight'], dtype=torch.float32, device=self.output_device)
                loss = nn.CrossEntropyLoss(**kwargs)
            else:
                loss = dynamic_import(loss_type)(**kwargs)
        return loss

    def load_pred(self, pred_type, **kwargs):
        if pred_type is None or pred_type == 'argmax':
            pred = functools.partial(torch.argmax, dim=1)
        elif pred_type == 'Identity':
            pred = lambda x: x
        else:
            pred = dynamic_import(pred_type)(**kwargs)
        return pred

    def inspect_control(self, epoch):
        try:
            with open(self.control_file_path, 'r') as f:
                control_dict = yaml.load(f, Loader=yaml.FullLoader)
            if 'save_model_next_epoch' in control_dict:
                if control_dict['save_model_next_epoch']:
                    torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, 'model_epoch%d.pt' % epoch))
                    print('saved model of epoch %d under %s' % (epoch, self.model_save_dir))
            control_dict['save_model_next_epoch'] = False
            if 'save_train_state' in control_dict:
                if control_dict['save_train_state']:
                    self.save_train_state(epoch)
                    print('saved train state of epoch %d under %s' % (epoch, self.model_save_dir))
            control_dict['save_train_state'] = False
            with open(self.control_file_path, 'w') as f:
                yaml.dump(control_dict, f)
        except:
            self.logger.error('fail to react to control!')

    def save_model(self, epoch=-1, model_index=None):
        if not isinstance(self.model, (list, tuple)):
            models = [self.model]
        else:
            models = self.model
        if model_index is None:
            model_index = list(range(len(models)))
        pt_filename = 'model_%d_epoch_{}.pt'.format(epoch)
        for index in model_index:
            model = models[index]
            torch.save(model.state_dict(), os.path.join(self.model_save_dir, pt_filename % index))
        self.logger.info('saved model of epoch %d under %s' % (epoch, self.model_save_dir))

    def init_train_state(self):
        if not isinstance(self.model, (list, tuple)):
            models = [self.model]
            optimizers = [self.optimizer]
        else:
            models = self.model
            optimizers = self.optimizer
        for index, (optimizer, model) in enumerate(zip(optimizers, models)):
            if self.train_state_dir is not None:
                self.start_epoch = self.from_train_state(model, optimizer, index)
            else:
                model.apply(init_weights)
                self.start_epoch = 0
            if isinstance(self.output_device, int):
                model.cuda(self.output_device)

    def run_train(self):
        self.init_train_state()
        control_dict = {'save_model_next_epoch': False}
        with open(self.control_file_path, 'w') as f:
            yaml.dump(control_dict, f)
        for epoch in range(self.start_epoch, self.epoch):
            if isinstance(self.model, (list, tuple)):
                for model in self.model:
                    model.train()
            else:
                self.model.train()
            self.train_epoch(epoch)
            if (epoch + 1) % self.eval_step == 0:
                with torch.no_grad():
                    if isinstance(self.model, (list, tuple)):
                        for model in self.model:
                            model.eval()
                    else:
                        self.model.eval()
                    self.eval_epoch(epoch)
            if (epoch + 1) % self.model_save_step == 0:
                self.save_model(epoch)
            # self.inspect_control(epoch)
        # if self.save_model_flag:
        #     torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, 'model_epoch%d.pt' % epoch))
        #     self.save_model_flag = False
        # if epoch == num_epoch - 1:
        #     plot_difference_distribution(epoch_output_list, epoch_label_list, 100)
        # finally save the model
        self.save_model(-1)
        self.save_properties()

    def iter_pass(self, data, label, other, extra=dict()):
        """
        Returns (output(predict probabilities or regression results), loss)
        """
        data = recursive_wrap_data(data, self.output_device)
        label = recursive_wrap_data(label, self.output_device)
        # print('other')
        # print(other)
        if self.train_type == 'rnn':
            state = None
            if state is None or True in other['clear_state']:
                # batch of subsequence
                # detach state across batches(subsequences)
                # Initialize `state` when either it is the first iteration or
                # using random sampling
                state = self.model.begin_state(batch_size=len(label), device=self.output_device)
            else:
                if not isinstance(state, tuple):
                    # `state` is a tensor for `nn.GRU`
                    # detach but does not clear state
                    state.detach_()
                else:
                    # `state` is a tuple of tensors for `nn.LSTM` and
                    # for our custom scratch implementation
                    for s in state:
                        s.detach_()
            # [batch_size, num_steps] -> [num_steps, batch_size] ->
            # [
            #   [1st in time series, 1st in batch],
            #   [1st in time series, 2nd in batch],
            #   ...
            #   [1st in time series, last in batch],
            #   [2nd in time series, 1st in batch],
            #   ...
            # ]
            label = label.reshape([-1])
            output, state = self.model(data, state, **extra)
        else:
            output = self.model(data, **extra)
        l = self.loss(output, label)
        return output, l, label

    def train_epoch(self, epoch):
        if self.train_type == 'gan':
            return self.train_epoch_gan(epoch)
        epoch_loss_list = []
        epoch_output_list = []
        epoch_label_list = []
        self.logger.info('train epoch: {}'.format(epoch + 1))
        for batch, (data, label, other) in enumerate(tqdm(self.train_iter)):
            self.optimizer.zero_grad()
            output, l, label = self.iter_pass(data, label, other, self.train_extra)

            # print('before backward')
            # print([param for param in self.model.parameters()])
            l.backward()
            if self.grad_alter_fn is not None:
                self.grad_alter_fn(self.model, **self.grad_alter_fn_arg)
            # print('after backward')
            # print([param for param in self.model.parameters()])
            # print('grad after backward')
            # print([param.grad for param in self.model.parameters()])
            # print([torch.nonzero(param.grad) for param in self.model.parameters()])
            self.optimizer.step()

            epoch_label_list.append(recursive_to_cpu(label))
            epoch_output_list.append(recursive_to_cpu(output))
            # print('batch %d' % batch)
            # print('l.item() %f' % l.item())
            epoch_loss_list.append(l.item())
        # print('len(epoch_output_list)')
        # print(len(epoch_output_list))
        epoch_output_list = self.output_collate_fn(epoch_output_list)
        # flatten batch dim
        epoch_label_list = torch.cat(epoch_label_list)
        # print('len(epoch_output_list)')
        # print(len(epoch_output_list))
        epoch_properties = dict()

        self.logger.debug('\tmean train loss: {}'.format(np.asarray(epoch_loss_list).mean()))
        self.writer.add_scalar('mean_train_loss', np.asarray(epoch_loss_list).mean(), epoch)
        self.train_loss_list.append(np.asarray(epoch_loss_list).mean())
        self.logger.debug('\tlearning rate: {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.writer.add_scalar('learning_rate', self.optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        self.learning_rate_list.append(self.optimizer.state_dict()['param_groups'][0]['lr'])

        if self.task_type == 'classification':
            epoch_pred_list = self.pred(epoch_output_list)
            # print('epoch_pred_list')
            # print(epoch_pred_list)
            # print('epoch_label_list')
            # print(epoch_label_list)
            epoch_acc = self.cal_epoch_acc(epoch_pred_list, epoch_label_list, epoch_output_list)
            self.train_pred_list.append(epoch_pred_list)
            self.logger.debug('\ttrain accuracy: {}'.format(epoch_acc))
            self.writer.add_scalar('train_accuracy', epoch_acc, epoch)
            self.train_accuracy_list.append(epoch_acc)
            self.logger.debug('\n' + str(self.cal_epoch_cm(epoch_pred_list, epoch_label_list, epoch_output_list)))

            epoch_properties['epoch_pred_list'] = epoch_pred_list

        epoch_properties['epoch_label_list'] = epoch_label_list
        epoch_properties['epoch_output_list'] = epoch_output_list

        # self.save_epoch_properties(epoch_properties, epoch, False)

        self.scheduler.step()

    def eval_epoch(self, epoch):
        if self.train_type == 'gan':
            return
            # return self.eval_epoch_gan(epoch)
        epoch_loss_list = []
        epoch_output_list = []
        epoch_label_list = []
        self.logger.info('eval epoch: {}'.format(epoch + 1))
        for batch, (data, label, other) in enumerate(tqdm(self.test_iter)):
            output, l, label = self.iter_pass(data, label, other, self.test_extra)

            epoch_label_list.append(recursive_to_cpu(label))
            epoch_output_list.append(recursive_to_cpu(output))
            epoch_loss_list.append(l.item())

        epoch_output_list = self.output_collate_fn(epoch_output_list)
        epoch_label_list = torch.cat(epoch_label_list)

        epoch_properties = dict()

        self.logger.debug('\tmean test loss: {}'.format(np.asarray(epoch_loss_list).mean()))
        self.eva_loss_list.append(np.asarray(epoch_loss_list).mean())
        self.writer.add_scalar('mean_test_loss', np.asarray(epoch_loss_list).mean(), epoch)

        if self.task_type == 'classification':
            epoch_pred_list = self.pred(epoch_output_list)
            epoch_acc = self.cal_epoch_acc(epoch_pred_list, epoch_label_list, epoch_output_list)
            self.test_pred_list.append(epoch_pred_list)
            self.logger.debug('\ttest accuracy: {}'.format(epoch_acc))
            self.writer.add_scalar('test_accuracy', epoch_acc, epoch)
            self.eval_accuracy_list.append(epoch_acc)
            self.logger.debug('\n' + str(self.cal_epoch_cm(epoch_pred_list, epoch_label_list, epoch_output_list)))

            epoch_properties['epoch_pred_list'] = epoch_pred_list

        epoch_properties['epoch_label_list'] = epoch_label_list
        epoch_properties['epoch_output_list'] = epoch_output_list

        # self.save_epoch_properties(epoch_properties, epoch, True)

    def save_epoch_properties(self, out_dict, epoch, is_test=True):
        for k, v in out_dict.items():
            file_name = 'test' + k + '_epoch%d.pkl' if is_test else 'train' + k + '_epoch%d.pkl'
            with open(os.path.join(self.log_dir, file_name % epoch), 'wb') as f:
                pickle.dump(v, f)

    def save_properties(self):
        if self.task_type == 'classification':
            with open(os.path.join(self.model_save_dir, 'train_pred_list.pkl'), 'wb') as f:
                pickle.dump(self.train_pred_list, f)
            with open(os.path.join(self.model_save_dir, 'test_pred_list.pkl'), 'wb') as f:
                pickle.dump(self.test_pred_list, f)
            with open(os.path.join(self.model_save_dir, 'train_accuracy_list.pkl'), 'wb') as f:
                pickle.dump(self.train_accuracy_list, f)
            with open(os.path.join(self.model_save_dir, 'eval_accuracy_list.pkl'), 'wb') as f:
                pickle.dump(self.eval_accuracy_list, f)
            print('train_accuracy_list\n', self.train_accuracy_list)
            print('eval_accuracy_list\n', self.eval_accuracy_list)

        with open(os.path.join(self.model_save_dir, 'train_loss_list.pkl'), 'wb') as f:
            pickle.dump(self.train_loss_list, f)
        with open(os.path.join(self.model_save_dir, 'eva_loss_list.pkl'), 'wb') as f:
            pickle.dump(self.eva_loss_list, f)
        with open(os.path.join(self.model_save_dir, 'learning_rate_list.pkl'), 'wb') as f:
            pickle.dump(self.learning_rate_list, f)
        print('train_loss_list\n', self.train_loss_list)
        print('eva_loss_list\n', self.eva_loss_list)
        print('learning_rate_list\n', self.learning_rate_list)

    def save_train_state(self, model, optimizer, epoch, index=None):
        state = {'epoch': epoch,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        pt_filename = 'train_state_%d.pt' % index if index is not None else 'train_state.pt'
        filename = os.path.join(self.train_state_dir, pt_filename)
        torch.save(state, filename)

    def from_train_state(self, model, optimizer, index=None):
        pt_filename = 'train_state_%d.pt' % index if index is not None else 'train_state.pt'
        filename = os.path.join(self.train_state_dir, pt_filename)
        state = torch.load(filename)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['model'])
        start_epoch = state['epoch']
        return start_epoch


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


class TrainWGAN(TrainGAN):
    def __init__(self, config_dict, train_type='classification', **kwargs):
        super(TrainWGAN, self).__init__(config_dict, train_type, **kwargs)

        if 'lambda_gp' in self.config_dict['train_arg']:
            self.lambda_gp = self.config_dict['train_arg']['lambda_gp']
        else:
            self.lambda_gp = None
        print('self.lambda_gp')
        print(self.lambda_gp)

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
        self.logger.info(str(np.where(sample[:, 0] > 0.5, 1, 0)[:32]))
        self.logger.info(str(np.where(sample[:, 1] > 0.5, 2, 0)[:32]))
        self.logger.info(str(sample[:, 2:][:16]))

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
            # print('dis_fake_cls_out')
            # print(dis_fake_cls_out)
            # total_acc += torch.sum(dis_fake_cls_out < 0.5).data.item()
            # print('dis_real_cls_out')
            # print(dis_real_cls_out)
            # total_acc += torch.sum(dis_real_cls_out > 0.5).data.item()
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
        print('self.lambda_gp')
        print(self.lambda_gp)

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

            for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
                batch_size = cond_data.shape[0]
                total_sample_num += batch_size
                cond_data = recursive_wrap_data(cond_data, self.output_device)
                real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)

                # train generator batch
                if batch in train_gen_batches or (epoch_gen_loss / total_sample_num) > 5:
                    gen_loss = self.train_generator_batch(batch, cond_data, real_gen_output, other)
                    epoch_gen_loss += gen_loss * batch_size

                # train discriminator batch
                if batch in train_dis_batches:
                    # if abs(win_avg_fake_loss) > 100 or abs(win_avg_real_loss) > 100 or (epoch_gen_loss / total_sample_num) > 10:
                    #     continue
                    loss, fake_loss, real_loss, gp_loss = self.train_discriminator_batch(batch, cond_data, real_gen_output, other)

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

            sample = gen(cond_data)[0].cpu().detach().numpy()
            self.logger.info(str(np.where(sample[:, 0] > 0.5, 1, 0)[:32]))
            self.logger.info(str(np.where(sample[:, 1] > 0.5, 2, 0)[:32]))
            self.logger.info('\n' + str(sample[:, 2:][:16]))

            avg_loss = total_loss / total_sample_num
            avg_fake_loss = epoch_fake_loss / total_sample_num
            avg_real_loss = epoch_real_loss / total_sample_num
            self.dis_loss_list.append(avg_loss)

            self.logger.info('avg_loss = %.8f' % avg_loss)
            self.logger.info('avg_fake_loss = %.8f' % avg_fake_loss)
            self.logger.info('avg_real_loss = %.8f' % avg_real_loss)
            if self.lambda_gp is not None:
                self.logger.info('avg_gp_loss = %.8f' % (epoch_gp_loss / len(self.train_iter)))

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

    def train_generator_batch(self, batch, cond_data, real_gen_output, other):
        """
        Using Wesserstein GAN Loss
        """
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

    def train_discriminator_batch(self, batch, cond_data, real_gen_output, other):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
        Samples are drawn d_steps times, and the discriminator is trained for 'epochs' epochs.
        """
        optimizer_D = self.optimizer[1]
        # loss_D = self.loss[1]
        gen, dis = self.model

        # real_gen_output_as_input = torch.cat([torch.zeros([batch_size, 1]), real_gen_output[:, :-1]], dim=1)
        optimizer_D.zero_grad()
        # real
        dis_real_cls_out = dis(cond_data, real_gen_output)
        with torch.no_grad():
            fake = gen(cond_data)

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


class TrainRNNGANPretrain(TrainGAN):
    def run_train(self):
        self.init_train_state()
        control_dict = {'save_model_next_epoch': False}
        with open(self.control_file_path, 'w') as f:
            yaml.dump(control_dict, f)

        # GENERATOR MLE TRAINING
        print('Starting Generator MLE Training...')
        for epoch in range(self.config_dict['train_arg']['generator_pretrain_epoch']):
            print('epoch %d : ' % (epoch + 1), end='')
            self.pretrain_generator()
        self.save_model(-16)

        # torch.save(gen.state_dict(), pretrained_gen_path)
        # gen.load_state_dict(torch.load(pretrained_gen_path))

        # PRETRAIN DISCRIMINATOR
        print('\nStarting Discriminator Training...')
        for epoch in range(self.config_dict['train_arg']['discriminator_pretrain_epoch']):
            print('epoch %d : ' % (epoch + 1), end='')
            self.train_discriminator()

        # torch.save(dis.state_dict(), pretrained_dis_path)
        # dis.load_state_dict(torch.load(pretrained_dis_path))

        # # ADVERSARIAL TRAINING
        print('\nStarting Adversarial Training...')

        for epoch in range(self.epoch):
            print('\n--------\nEPOCH %d\n--------' % (epoch + 1))
            if self.config_dict['train_arg']['adaptive_adv_train']:
                while self.train_generator() > 0.6:
                    pass
                while self.train_discriminator() < 0.6:
                    pass
            else:
                # TRAIN GENERATOR
                print('\nAdversarial Training Generator : ', end='')
                for i in range(self.config_dict['train_arg']['adv_generator_epoch']):
                    self.train_generator()

                # TRAIN DISCRIMINATOR
                print('\nAdversarial Training Discriminator : ')
                for i in range(self.config_dict['train_arg']['adv_discriminator_epoch']):
                    self.train_discriminator()
                # if (epoch + 1) % self.model_save_step == 0:
            self.save_model(epoch, (0, ))

        self.save_model(-1)
        # self.save_properties()

    def pretrain_generator(self):
        """
        Max Likelihood Pretraining for the generator
        """
        optimizer_G = self.optimizer[0]
        # loss_G, loss_D = self.loss
        gen = self.model[0]
        total_loss = 0
        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            batch_size = cond_data.shape[0]
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            # print(real_gen_output)
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            real_gen_output_as_input = torch.cat([torch.zeros([batch_size, 1], device=cond_data.device, dtype=torch.long), real_gen_output[:, :-1]], dim=1)

            optimizer_G.zero_grad()
            loss, h_gen = gen.batchNLLLoss(cond_data, real_gen_output_as_input, real_gen_output)
            loss.backward()
            optimizer_G.step()

            total_loss += loss.data.item()

        sys.stdout.flush()
        print('gen.sample(5)')
        print(gen.sample(cond_data)[0][:1].cpu().detach().numpy().tolist())
        # each loss in a batch is loss per sample
        total_loss = total_loss / len(self.train_iter.dataset)
        # # sample from generator and compute oracle NLL
        # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
        #                                            start_letter=START_LETTER, gpu=CUDA)

        print(' average_train_NLL = %.4f' % total_loss)
        # print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))

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
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            real_gen_output_as_input = torch.cat([torch.zeros([batch_size, 1], device=cond_data.device, dtype=torch.long), real_gen_output[:, :-1]], dim=1)

            fake, h_gen = gen.sample(cond_data)
            rewards, h_dis = dis.batchClassify(cond_data, fake)
            epoch_dis_acc += torch.sum(rewards < 0.5).data.item()

            optimizer_G.zero_grad()
            pg_loss, h_gen_PG = gen.batchPGLoss(cond_data, real_gen_output_as_input, real_gen_output, rewards)
            epoch_pg_loss += pg_loss.item()
            pg_loss.backward()
            optimizer_G.step()

        epoch_dis_acc = epoch_dis_acc / total_sample_num
        print('epoch_dis_acc %.3f' % epoch_dis_acc)

        sys.stdout.flush()
        print('gen.sample(5)')
        print(gen.sample(cond_data)[0].cpu().detach().numpy().tolist())

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

        print(fake[0].cpu().detach().numpy().tolist())
        print(real_gen_output[0].cpu().detach().numpy().tolist())

        avg_loss = total_loss / total_sample_num
        avg_acc = total_acc / total_sample_num / 2

        sys.stdout.flush()
        print(' average_loss = %.4f, train_acc = %.4f' % (
            avg_loss, avg_acc))

        return avg_acc


class TrainSeqGANAdvLoss(TrainRNNGANPretrain):
    def pretrain_generator(self):
        """
        Max Likelihood Pretraining for the generator
        """
        optimizer_G = self.optimizer[0]
        # loss_G, loss_D = self.loss
        gen = self.model[0]
        total_loss = 0
        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            batch_size = cond_data.shape[0]
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            # print(real_gen_output)
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            real_gen_output_as_input = torch.cat(
                [torch.tensor([[[1, 0.5, 0.5]] for _ in range(batch_size)], device=cond_data.device, dtype=torch.float),
                 real_gen_output[:, :-1]], dim=1
            )

            optimizer_G.zero_grad()
            loss, h_gen = gen.batchNLLLoss(cond_data, real_gen_output_as_input, real_gen_output)
            loss.backward()
            optimizer_G.step()

            total_loss += loss.data.item()

        sys.stdout.flush()
        print('gen.sample(5)')
        label, pos = gen.sample(cond_data)[0]
        print(label[0].cpu().detach().numpy().tolist())
        print(pos[0].cpu().detach().numpy().tolist()[:15])
        # each loss in a batch is loss per sample
        total_loss = total_loss / len(self.train_iter.dataset)
        # # sample from generator and compute oracle NLL
        # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
        #                                            start_letter=START_LETTER, gpu=CUDA)

        print(' average_train_loss = %.4f' % total_loss)
        # print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))
        return total_loss

    def train_generator(self):
        """
        The generator is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        optimizer_G, optimizer_D = self.optimizer[1], self.optimizer[2]
        loss_G = self.loss[0]
        gen, dis = self.model
        epoch_pg_loss, epoch_adv_loss = 0, 0
        epoch_dis_acc = 0
        total_sample_num = 0

        sys.stdout.flush()
        for batch, (cond_data, real_gen_output, other) in enumerate(tqdm(self.train_iter)):
            batch_size = cond_data.shape[0]
            total_sample_num += batch_size
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            real_gen_output_as_input = torch.cat(
                [torch.tensor([[[1, 0.5, 0.5]] for _ in range(batch_size)], device=cond_data.device, dtype=torch.float),
                 real_gen_output[:, :-1]], dim=1
            )
            # print(real_gen_output)
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            fake, h_gen = gen.sample(cond_data)
            rewards, h_dis = dis.batchClassify(cond_data, fake)
            epoch_dis_acc += torch.sum(rewards < 0.5).data.item()
            # print('rewards')
            # print(rewards)
            adv_loss = loss_G(rewards, torch.ones(batch_size, device=cond_data.device))
            # print('adv_loss')
            # print(adv_loss)
            if 'adv_loss_multiplier' in self.config_dict['train_arg']:
                adv_loss_multiplier = self.config_dict['train_arg']['adv_loss_multiplier']
                if adv_loss_multiplier is not None:
                    adv_loss *= adv_loss_multiplier
            epoch_adv_loss += adv_loss.item()
            # print('adv_loss')
            # print(adv_loss)

            pg_loss, h_gen_PG = gen.batchPGLoss(cond_data, real_gen_output_as_input, real_gen_output, rewards)
            epoch_pg_loss += pg_loss.item()

            (pg_loss + adv_loss).backward()
            # if self.grad_alter_fn is not None:
            #     self.grad_alter_fn(gen.parameters())
            # print('gen.gru2out_pos[0].weight.grad')
            # print(gen.gru2out_pos[0].weight.grad)
            optimizer_G.step()

        sys.stdout.flush()
        print('pg_loss %.3f' % (epoch_pg_loss / len(self.train_iter)))
        print('adv_loss %.3f' % (epoch_adv_loss / len(self.train_iter)))
        epoch_dis_acc = epoch_dis_acc / total_sample_num
        print('epoch_dis_acc %.3f' % epoch_dis_acc)
        print('gen.sample(5)')
        label, pos = gen.sample(cond_data)[0]
        print(label[0].cpu().detach().numpy().tolist())
        # print(pos[0].cpu().detach().numpy().tolist()[:15])

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
            if random.random() < 0.1:
                continue
            batch_size = cond_data.shape[0]
            total_sample_num += batch_size
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            # real_gen_output_as_input = torch.cat([torch.zeros([batch_size, 1]), real_gen_output[:, :-1]], dim=1)
            optimizer_D.zero_grad()

            fake, h_gen = gen.sample(cond_data)
            fake = [fake[0], fake[1].detach()]

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

        avg_loss = total_loss / total_sample_num
        avg_acc = total_acc / total_sample_num / 2

        sys.stdout.flush()
        print(' average_loss = %.4f, train_acc = %.4f' % (
            avg_loss, avg_acc))

        return avg_acc


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


def recursive_detach(items):
    if isinstance(items, torch.Tensor):
        return items.detach()
    elif isinstance(items, list):
        for i, item in enumerate(items):
            items[i] = recursive_detach(item)
    else:
        return items


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)
        torch.nn.init.normal_(m.bias, std=0.01)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight, std=0.01)
        torch.nn.init.normal_(m.bias, std=0.01)


def plot_difference_distribution(pred_list, target_list, difference_step):
    difference_list = []
    for i in range(len(pred_list)):
        difference_list.append(pred_list[i] - target_list[i])
    max_pos_difference = np.max(difference_list)
    max_neg_difference = np.min(difference_list)
    section_interval = (max_pos_difference - max_neg_difference) / difference_step
    y = np.zeros(difference_step)
    for i in range(len(difference_list)):
        section_index = math.floor((difference_list[i] - max_neg_difference) / section_interval)
        if section_index == difference_step:
            section_index -= 1
        y[section_index] += 1
    x = list(range(difference_step))
    for i in range(difference_step):
        x[i] = x[i] * section_interval + max_neg_difference + section_interval / 2
    plt.plot(x, y)
    plt.show()


def train_with_config(config_path, format_config=False, folds=5):
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    for fold in range(1, folds + 1):
        print('Fold %d' % fold)
        if format_config:
            formatted_config_dict = get_fold_config(config_dict, fold)
        else:
            formatted_config_dict = config_dict
        if formatted_config_dict['train_type'] == 'rnngan_with_pretrain':
            train = TrainRNNGANPretrain(formatted_config_dict)
        elif formatted_config_dict['train_type'] == 'gan':
            train = TrainGAN(formatted_config_dict)
        elif formatted_config_dict['train_type'] == 'wgan':
            train = TrainWGAN(formatted_config_dict)
        elif formatted_config_dict['train_type'] == 'wganwb':
            train = TrainWGANWithinBatch(formatted_config_dict)
        elif formatted_config_dict['train_type'] == 'vae':
            train = TrainVAE(formatted_config_dict)
        elif formatted_config_dict['train_type'] == 'seq2seq':
            train = TrainSeq2Seq(formatted_config_dict)
        # elif formatted_config_dict['train_type'] == 'rnngan_with_pretrain_inherit':
        #     train = TrainRNNGANPretrainInherit(formatted_config_dict)
        elif formatted_config_dict['train_type'] == 'seqgan_adv_loss':
            train = TrainSeqGANAdvLoss(formatted_config_dict)
        else:
            train = Train(formatted_config_dict)
        train.run_train()


def get_fold_config(config_dict, fold):
    fold_config_dict = copy.deepcopy(config_dict)
    # possible items which needs formatting with fold
    for path in [
        ['data_arg', 'test_dataset_arg', 'table_name'],
        ['data_arg', 'test_dataset_arg', 'data_path'],
        ['data_arg', 'test_dataset_arg', 'label_path'],
        ['data_arg', 'train_dataset_arg', 'table_name'],
        ['data_arg', 'train_dataset_arg', 'data_path'],
        ['data_arg', 'train_dataset_arg', 'label_path'],
        ['output_arg', 'log_dir'],
        ['output_arg', 'model_save_dir'],
    ]:
        try_format_dict_with_path(fold_config_dict, path, fold)
    # print(fold_config_dict)
    return fold_config_dict
