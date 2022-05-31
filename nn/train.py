import functools
import logging
import math
import os
import pickle
import threading
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils import data
from tqdm import tqdm

from util.general_util import dynamic_import, recursive_to_cpu, recursive_wrap_data, try_format_dict_with_path
from nn.metrics import default_metrices
from nn.dataset import collate_fn

np.set_printoptions(suppress=True)


class Train:
    def __init__(self,
                 model_arg, optimizer_arg, scheduler_arg, data_arg, loss_arg, pred_arg, output_arg, train_arg,
                 task_type='classification',
                 **kwargs):
        self.task_type = task_type
        if 'rnn' in kwargs:
            self.rnn = kwargs['rnn']
            # self.use_random_iter = kwargs['use_random_iter']
        else:
            self.rnn = False

        if self.task_type == 'classification':
            if 'cal_acc_func' in kwargs and kwargs['cal_acc_func'] is not None:
                self.cal_epoch_acc = dynamic_import(kwargs['cal_acc_func'])
            else:
                self.cal_epoch_acc = default_metrices.cal_acc_func
            if 'cal_cm_func' in kwargs and kwargs['cal_cm_func'] is not None:
                self.cal_epoch_cm = dynamic_import(kwargs['cal_cm_func'])
            else:
                self.cal_epoch_cm = default_metrices.cal_cm_func
            self.pred = self.load_pred(**pred_arg)
            self.train_pred_list = []
            self.test_pred_list = []
            self.train_accuracy_list = []
            self.eval_accuracy_list = []

        if 'collate_fn' in kwargs and kwargs['collate_fn'] is not None:
            self.collate_fn = dynamic_import(kwargs['collate_fn'])
        else:
            self.collate_fn = None
        if 'output_collate_fn' in kwargs and kwargs['output_collate_fn'] is not None:
            self.output_collate_fn = dynamic_import(kwargs['output_collate_fn'])
        else:
            self.output_collate_fn = collate_fn.output_collate_fn
        if 'output_device' in kwargs and kwargs['output_device'] is not None:
            self.output_device = kwargs['output_device']
        else:
            self.output_device = 'cpu'
        if 'grad_alter_fn' in kwargs and kwargs['grad_alter_fn'] is not None:
            self.grad_alter_fn = dynamic_import(kwargs['grad_alter_fn'])
            self.grad_alter_fn_arg = kwargs['grad_alter_fn_arg']
        else:
            self.grad_alter_fn = None

        self.train_state_dir = kwargs['train_state_dir'] if 'train_state_dir' in kwargs else None

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
        if model_type == 'mlp':
            net = nn.Sequential(
                nn.Linear(9, 64),
                nn.ReLU(),
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1))
        else:
            net = dynamic_import(model_type)(**kwargs)
        self.logger.info('loaded model: %s' % model_type)
        return net

    def load_optimizer(self, optimizer_type, **kwargs):
        if optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), **kwargs)
        else:
            optimizer = dynamic_import(optimizer_type)(**kwargs)
        return optimizer

    def load_scheduler(self, scheduler_type, **kwargs):
        if scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **kwargs)
        else:
            scheduler = dynamic_import(scheduler_type)(**kwargs)
        return scheduler

    def load_data(self,
                  dataset,
                  train_dataset_arg,
                  test_dataset_arg,
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
        test_iter = torch.utils.data.DataLoader(
            dataset=dynamic_import(dataset)(**test_dataset_arg),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            drop_last=False)
        return train_iter, test_iter

    def load_loss(self, loss_type, **kwargs):
        if loss_type == 'MSE':
            loss = nn.MSELoss(**kwargs)
        elif loss_type == 'CrossEntropy':
            loss = nn.CrossEntropyLoss(**kwargs)
        else:
            loss = dynamic_import(loss_type)(**kwargs)
        return loss

    def load_pred(self, pred_type, **kwargs):
        if pred_type is None or pred_type == 'argmax':
            pred = functools.partial(torch.argmax, dim=1)
        else:
            pred = dynamic_import(pred_type)(**kwargs)
        return pred

    def run_train_thread(self, num_epoch, eval_step):
        class TrainThread(threading.Thread):
            def __init__(self, obj, num_epoch, eval_step):
                self.obj = obj
                self.num_epoch, self.eval_step = num_epoch, eval_step
                super().__init__()

            def run(self):
                self.obj.run_train(self.num_epoch, self.eval_step)

        train_thread = TrainThread(self, num_epoch, eval_step)
        train_thread.start()
        while True:
            if input('save_model?') == 'y':
                self.save_model_flag = True
            if not train_thread.is_alive():
                return

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

    def run_train(self):
        if self.train_state_dir is not None:
            self.from_train_state()
        else:
            self.model.apply(init_weights)
            self.start_epoch = 0
        if isinstance(self.output_device, int):
            self.model.cuda(self.output_device)

        control_dict = {'save_model_next_epoch': False}
        with open(self.control_file_path, 'w') as f:
            yaml.dump(control_dict, f)
        for epoch in range(self.start_epoch, self.epoch):
            self.train_epoch(epoch)
            if (epoch + 1) % self.eval_step == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.eval_epoch(epoch)
                    self.model.train()
            if (epoch + 1) % self.model_save_step == 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, 'model_epoch%d.pt' % epoch))
                print('saved model of epoch %d under %s' % (epoch, self.model_save_dir))
            self.inspect_control(epoch)
        # if self.save_model_flag:
        #     torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, 'model_epoch%d.pt' % epoch))
        #     self.save_model_flag = False
        # if epoch == num_epoch - 1:
        #     plot_difference_distribution(epoch_output_list, epoch_label_list, 100)
        # finally save the model
        torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, 'model_final.pt'))
        self.save_properties()

    def iter_pass(self, data, label, other):
        """
        Returns (output(predict probabilities or regression results), loss)
        """
        data = recursive_wrap_data(data, self.output_device)
        label = recursive_wrap_data(label, self.output_device)
        if self.rnn:
            state = None
            if state is None or other['clear_state']:
                # batch of subsequence
                # detach state across batches(subsequences)
                # Initialize `state` when either it is the first iteration or
                # using random sampling
                state = self.model.begin_state(batch_size=len(data[0]), device=data[0].device)
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
            output, state = self.model(data, state)
        else:
            output = self.model(data)
        # print('output.shape')
        # print(output.shape)
        # print('label')
        # print(label)
        l = self.loss(output, label)
        return output, l, label

    def train_epoch(self, epoch):
        epoch_loss_list = []
        epoch_output_list = []
        epoch_label_list = []
        self.logger.info('train epoch: {}'.format(epoch + 1))
        for batch, (data, label, other) in enumerate(tqdm(self.train_iter)):
            self.optimizer.zero_grad()
            output, l, label = self.iter_pass(data, label, other)

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
            epoch_loss_list.append(l.item())
        # print('len(epoch_output_list)')
        # print(len(epoch_output_list))
        epoch_output_list = self.output_collate_fn(epoch_output_list)
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

        self.save_epoch_properties(epoch_properties, epoch, False)

        self.scheduler.step()

    def eval_epoch(self, epoch):
        epoch_loss_list = []
        epoch_output_list = []
        epoch_label_list = []
        self.logger.info('eval epoch: {}'.format(epoch + 1))
        for batch, (data, label, other) in enumerate(tqdm(self.test_iter)):
            output, l, label = self.iter_pass(data, label, other)

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

        self.save_epoch_properties(epoch_properties, epoch, True)

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

    def save_train_state(self, epoch):
        state = {'epoch': epoch,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        filename = os.path.join(self.train_state_dir, 'train_state.pt')
        torch.save(state, filename)

    def from_train_state(self):
        filename = os.path.join(self.train_state_dir, 'train_state.pt')
        state = torch.load(filename)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['model'])
        self.start_epoch = state['epoch']


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
        train = Train(**formatted_config_dict)
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
