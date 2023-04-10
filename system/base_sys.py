import functools
import logging
import os
import pickle

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils import data
from tqdm import tqdm

from nn.dataset import collate_fn
from nn.metrics import default_metrices
from util.general_util import dynamic_import, recursive_to_cpu, recursive_wrap_data


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
