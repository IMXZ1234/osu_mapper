import math
import numpy as np
import torch
import os
import pickle
import logging
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from torch.utils import data
import yaml
import matplotlib.pyplot as plt
import threading
import functools

from util.general_util import dynamic_import
from util.result import metrics_util
np.set_printoptions(suppress=True)


class Train:
    def __init__(self, model_arg, optimizer_arg, scheduler_arg, data_arg, loss_arg, pred_arg, output_arg, train_arg,
                 **kwargs):
        if 'cal_acc_func' in kwargs:
            self.cal_epoch_acc = dynamic_import(kwargs['cal_acc_func'])
        else:
            self.cal_epoch_acc = self.cal_acc_func
        if 'cal_cm_func' in kwargs:
            self.cal_epoch_cm = dynamic_import(kwargs['cal_cm_func'])
        else:
            self.cal_epoch_cm = self.cal_cm_func
        if 'collate_fn' in kwargs:
            self.collate_fn = dynamic_import(kwargs['collate_fn'])
        else:
            self.collate_fn = None
        if 'output_device' in kwargs:
            self.output_device = kwargs['output_device']
        else:
            self.output_device = 'cpu'

        self.train_state_dir = kwargs['train_state_dir'] if 'train_state_dir' in kwargs else None

        self.start_epoch = 0
        self.writer, self.logger, self.log_dir, self.model_save_dir, self.model_save_step = self.load_logger(
            **output_arg)
        self.model = self.load_model(**model_arg)
        self.optimizer = self.load_optimizer(**optimizer_arg)
        self.scheduler = self.load_scheduler(**scheduler_arg)
        self.train_iter, self.test_iter = self.load_data(**data_arg)
        self.loss = self.load_loss(**loss_arg)
        self.pred = self.load_pred(**pred_arg)

        self.epoch = train_arg['epoch']
        self.eval_step = train_arg['eval_step']

        self.control_file_path = os.path.join(self.log_dir, 'control.yaml')
        self.train_pred_list = []
        self.test_pred_list = []
        self.train_accuracy_list = []
        self.train_loss_list = []
        self.eval_accuracy_list = []
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
                  drop_last=False):
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
        if pred_type == 'classification':
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
        self.save_loss_acc()

    def train_epoch(self, epoch):
        epoch_loss_list = []
        epoch_output_list = []
        epoch_label_list = []
        epoch_index_list = []
        self.logger.info('train epoch: {}'.format(epoch + 1))
        for batch, (data, label, index) in enumerate(self.train_iter):
            epoch_index_list.append(index)
            data = recursive_wrap_data(data, self.output_device)
            label = recursive_wrap_data(label, self.output_device)
            output = self.model(data)
            l = self.loss(output, label)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            epoch_label_list.extend(recursive_to_cpu(label))
            epoch_output_list.append(recursive_to_cpu(output))
            epoch_loss_list.append(l.item())
        # print('len(epoch_output_list)')
        # print(len(epoch_output_list))
        epoch_output_list = self.output_collate_fn(epoch_output_list)
        # print('len(epoch_output_list)')
        # print(len(epoch_output_list))
        epoch_pred_list = self.pred(epoch_output_list)
        epoch_acc = self.cal_epoch_acc(epoch_pred_list, epoch_label_list, epoch_output_list)
        self.train_pred_list.append(epoch_pred_list)
        self.logger.debug('\ttrain accuracy: {}'.format(epoch_acc))
        self.writer.add_scalar('train_accuracy', epoch_acc, epoch)
        self.train_accuracy_list.append(epoch_acc)
        self.logger.debug('\n' + str(self.cal_epoch_cm(epoch_pred_list, epoch_label_list, epoch_output_list)))
        self.logger.debug('\tmean train loss: {}'.format(np.asarray(epoch_loss_list).mean()))
        self.writer.add_scalar('mean_train_loss', np.asarray(epoch_loss_list).mean(), epoch)
        self.train_loss_list.append(np.asarray(epoch_loss_list).mean())
        self.logger.debug('\tlearning rate: {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.writer.add_scalar('learning_rate', self.optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        self.learning_rate_list.append(self.optimizer.state_dict()['param_groups'][0]['lr'])
        self.scheduler.step()

        self.save_epoch_output({
            'epoch_pred_list': epoch_pred_list,
            'epoch_label_list': epoch_label_list,
            'epoch_output_list': epoch_output_list,
        }, epoch, False)

    def eval_epoch(self, epoch):
        epoch_loss_list = []
        epoch_output_list = []
        epoch_label_list = []
        epoch_index_list = []
        self.logger.info('eval epoch: {}'.format(epoch + 1))
        for batch, (data, label, index) in enumerate(self.test_iter):
            epoch_index_list.append(index)
            data = recursive_wrap_data(data, self.output_device)
            label = recursive_wrap_data(label, self.output_device)
            output = self.model(data)
            l = self.loss(output, label)
            epoch_label_list.extend(recursive_to_cpu(label))
            epoch_output_list.append(recursive_to_cpu(output))
            epoch_loss_list.append(l.item())
        # print('len(epoch_output_list)')
        # print(len(epoch_output_list))
        epoch_output_list = self.output_collate_fn(epoch_output_list)
        # print('len(epoch_output_list)')
        # print(len(epoch_output_list))
        epoch_pred_list = self.pred(epoch_output_list)
        epoch_acc = self.cal_epoch_acc(epoch_pred_list, epoch_label_list, epoch_output_list)
        self.test_pred_list.append(epoch_pred_list)
        self.logger.debug('\ttest accuracy: {}'.format(epoch_acc))
        self.writer.add_scalar('test_accuracy', epoch_acc, epoch)
        self.eval_accuracy_list.append(epoch_acc)
        self.logger.debug('\n' + str(self.cal_epoch_cm(epoch_pred_list, epoch_label_list, epoch_output_list)))
        self.logger.debug('\tmean test loss: {}'.format(np.asarray(epoch_loss_list).mean()))
        self.eva_loss_list.append(np.asarray(epoch_loss_list).mean())
        self.writer.add_scalar('mean_test_loss', np.asarray(epoch_loss_list).mean(), epoch)
        self.save_epoch_output({
            'epoch_pred_list': epoch_pred_list,
            'epoch_label_list': epoch_label_list,
            'epoch_output_list': epoch_output_list,
        }, epoch, True)

    @staticmethod
    def cal_acc_func(pred, label, output):
        """
        input should be tensors
        """
        return len(torch.where(pred == label)[0]) / len(label)

    @staticmethod
    def cal_cm_func(pred, label, output):
        return metrics_util.get_epoch_confusion(pred.numpy(), label.numpy())

    def save_epoch_output(self, out_dict, epoch, is_test=True):
        for k, v in out_dict.items():
            file_name = 'test' + k + '_epoch%d.pkl' if is_test else 'train' + k + '_epoch%d.pkl'
            with open(os.path.join(self.log_dir, file_name % epoch), 'wb') as f:
                pickle.dump(v, f)

    def output_collate_fn(self, epoch_output_list):
        """
        Flatten batches.
        """
        if isinstance(epoch_output_list[0], torch.Tensor):
            # if model output is a single tensor
            return torch.cat(epoch_output_list)
        # else model output is a list
        batch_out_item_num = len(epoch_output_list[0])
        collated = [[] for _ in range(batch_out_item_num)]
        for batch_idx, batch in enumerate(epoch_output_list):
            for item_idx in range(batch_out_item_num):
                if isinstance(batch[item_idx], (list, tuple)):
                    collated[item_idx].extend(batch[item_idx])
                else:
                    # a single item, most likely a tensor, with first dim as sample dim
                    collated[item_idx].append(batch[item_idx])
        for i in range(batch_out_item_num):
            if isinstance(collated[i][0], torch.Tensor):
                # concatenate tensors at sample dim(the first dim)
                collated[i] = torch.cat(collated[i])
        return collated

    def save_loss_acc(self):
        with open(os.path.join(self.model_save_dir, 'train_pred_list.pkl'), 'wb') as f:
            pickle.dump(self.train_pred_list, f)
        with open(os.path.join(self.model_save_dir, 'test_pred_list.pkl'), 'wb') as f:
            pickle.dump(self.test_pred_list, f)
        with open(os.path.join(self.model_save_dir, 'train_accuracy_list.pkl'), 'wb') as f:
            pickle.dump(self.train_accuracy_list, f)
        with open(os.path.join(self.model_save_dir, 'train_loss_list.pkl'), 'wb') as f:
            pickle.dump(self.train_loss_list, f)
        with open(os.path.join(self.model_save_dir, 'eval_accuracy_list.pkl'), 'wb') as f:
            pickle.dump(self.eval_accuracy_list, f)
        with open(os.path.join(self.model_save_dir, 'eva_loss_list.pkl'), 'wb') as f:
            pickle.dump(self.eva_loss_list, f)
        with open(os.path.join(self.model_save_dir, 'learning_rate_list.pkl'), 'wb') as f:
            pickle.dump(self.learning_rate_list, f)
        print('train_accuracy_list\n', self.train_accuracy_list)
        print('train_loss_list\n', self.train_loss_list)
        print('eval_accuracy_list\n', self.eval_accuracy_list)
        print('eva_loss_list\n', self.eva_loss_list)
        print('learning_rate_list\n', self.learning_rate_list)
        # plt.plot(self.train_accuracy_list)
        # plt.show()
        # plt.plot(self.train_loss_list)
        # plt.show()
        # plt.plot(self.eval_accuracy_list)
        # plt.show()
        # plt.plot(self.eva_loss_list)
        # plt.show()
        # plt.plot(self.learning_rate_list)
        # plt.show()

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
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight, std=0.01)
        # torch.nn.init.normal_(m.bias, std=0.01)


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


def recursive_to_cpu(value):
    """
    Shallow copy here for lists.

    :param value:
    :return:
    """
    if isinstance(value, torch.Tensor):
        # most features generated by network is gpu tensor
        # print('value is tensor')
        return value.cpu().detach()
    elif isinstance(value, list):
        # if is list of gpu tensor
        # print('value is list')
        for i in range(len(value)):
            value[i] = recursive_to_cpu(value[i])
    elif isinstance(value, tuple):
        value = list(value)
        for i in range(len(value)):
            value[i] = recursive_to_cpu(value[i])
    return value


def recursive_wrap_data(data, output_device):
    """
    recursively wrap tensors in data into Variable and move to device.

    :param data:
    :return:
    """
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = recursive_wrap_data(data[i], output_device)
    elif isinstance(data, torch.Tensor):
        return Variable(data.cuda(output_device), requires_grad=False)
    return data


def train_with_new_setting(setting_name):
    for lr in [0.1]:
        print('init lr %s' % str(lr))
        config_path = './resources/config/%s.yaml' % setting_name
        model_arg = {'model_type': 'net.seg_multi_pred_mlp.SegMultiPredMLP',
                     'num_classes': 2,
                     'extract_hidden_layer_num': 0,
                     'beat_feature_frames': 16384,
                     'sample_beats': 16,
                     'pad_beats': 4,
                     }  # , 'num_block': [1, 1, 1, 1]
        optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
        scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 3, 'gamma': 0.1}
        data_arg = {'dataset': 'dataset.seg_multi_label_db_dataset.SegMultiLabelDBDataset',
                    'train_dataset_arg':
                        {'db_path': r'./resources/data/osu_train.db',
                         'audio_dir': r'./resources/data/audio',
                         'table_name': 'TRAINFOLD%d',
                         'beat_feature_frames': 16384,
                         # this 8+32+8 will yield an audio fragment of about 17 sec
                         'sample_beats': 16,
                         'pad_beats': 4,
                         'multi_label': False, },
                    'test_dataset_arg':
                        {'db_path': r'./resources/data/osu_train.db',
                         'audio_dir': r'./resources/data/audio',
                         'table_name': 'TESTFOLD%d',
                         'beat_feature_frames': 16384,
                         'sample_beats': 16,
                         'pad_beats': 4,
                         'multi_label': False, },
                    'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 1,
                    'drop_last': False}
        loss_arg = {'loss_type': 'loss.multi_pred_loss.MultiPredLoss'}
        pred_arg = {'pred_type': 'pred.multi_pred.MultiPred'}
        output_arg = {'log_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_dir': './resources/result/' + setting_name + '/' + str(lr) + '/%d',
                      'model_save_step': 1}
        train_arg = {'epoch': 24, 'eval_step': 1}
        with open(config_path, 'w') as f:
            yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                       'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                       'train_arg': train_arg,
                       'output_device': 0,
                       'collate_fn': 'dataset.collate_fn.data_array_to_tensor',
                       'cal_acc_func': 'metrics.multi_pred_metrics.multi_pred_cal_acc_func',
                       'cal_cm_func': 'metrics.multi_pred_metrics.multi_pred_cal_cm_func',
                       }, f)
        train_with_config(config_path, folds=5)


def train_with_config(config_path, folds=5):
    for fold in range(1, folds + 1):
        with open(config_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        print('Fold %d' % fold)
        get_fold_config(config_dict, fold)
        train = Train(**config_dict)
        train.run_train()


def get_fold_config(config_dict, fold):
    fold_config_dict = config_dict.copy()
    fold_config_dict['data_arg']['test_dataset_arg']['table_name'] = \
        fold_config_dict['data_arg']['test_dataset_arg']['table_name'] % fold
    fold_config_dict['data_arg']['train_dataset_arg']['table_name'] = \
        fold_config_dict['data_arg']['train_dataset_arg']['table_name'] % fold
    fold_config_dict['output_arg']['log_dir'] = \
        fold_config_dict['output_arg']['log_dir'] % fold
    fold_config_dict['output_arg']['model_save_dir'] = \
        fold_config_dict['output_arg']['model_save_dir'] % fold
    return fold_config_dict


if __name__ == '__main__':
    setting_name = 'seg_mlp_bi_lr0.1'
    train_with_new_setting(setting_name)
    # config_path = './resources/config/%s.yaml' % setting_name
    # with open(config_path, 'r') as f:
    #     config_dict = yaml.load(f, Loader=yaml.FullLoader)
    # for lr in [0.1]:
    #     print('init lr %s' % str(lr))
    #     for fold in range(1, 6):
    #         print('Fold %d' % fold)
    #         fold_config_dict = get_fold_config(config_dict, fold)
    #         train = Train(**fold_config_dict)
    #         train.run_train()
