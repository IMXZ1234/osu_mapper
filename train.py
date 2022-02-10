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


class Train:
    def __init__(self, model_arg, optimizer_arg, scheduler_arg, data_arg, loss_arg, pred_arg, output_arg, **kwargs):
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
        self.model = self.load_model(**model_arg)
        self.optimizer = self.load_optimizer(**optimizer_arg)
        self.scheduler = self.load_scheduler(**scheduler_arg)
        self.train_iter, self.test_iter = self.load_data(**data_arg)
        self.loss = self.load_loss(**loss_arg)
        self.pred = self.load_pred(**pred_arg)
        self.writer, self.logger, self.log_dir, self.model_save_dir = self.load_logger(**output_arg)
        self.control_file_path = os.path.join(self.log_dir, 'control.yaml')
        self.train_pred_list = []
        self.test_pred_list = []
        self.train_accuracy_list = []
        self.train_loss_list = []
        self.eval_accuracy_list = []
        self.eva_loss_list = []
        self.learning_rate_list = []
        self.save_model_flag = False

    def load_logger(self, log_dir, model_save_dir):
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
        return writer, logger, log_dir, model_save_dir

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
        net.apply(init_weights)
        if isinstance(self.output_device, int):
            net.cuda(self.output_device)
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
        else:
            loss = dynamic_import(loss_type)(**kwargs)
        return loss

    def load_pred(self, pred_type, **kwargs):
        pred = dynamic_import(pred_type)(**kwargs)
        return pred

    def run_train(self, num_epoch, eval_step):
        class TrainThread(threading.Thread):
            def __init__(self, obj, num_epoch, eval_step):
                self.obj = obj
                self.num_epoch, self.eval_step = num_epoch, eval_step
                super().__init__()

            def run(self):
                self.obj.train(self.num_epoch, self.eval_step)
        train_thread = TrainThread(self, num_epoch, eval_step)
        train_thread.start()
        while True:
            if input('save_model?') == 'y':
                self.save_model_flag = True
            if not train_thread.is_alive():
                return

    def train(self, num_epoch, eval_step):
        control_dict = {'save_model_next_epoch': False}
        with open(self.control_file_path, 'w') as f:
            yaml.dump(control_dict, f)
        for epoch in range(num_epoch):
            self.train_epoch(epoch)
            if (epoch + 1) % eval_step == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.eval_epoch(epoch)
                    self.model.train()
            with open(self.control_file_path, 'r') as f:
                control_dict = yaml.load(f, Loader=yaml.FullLoader)
            if control_dict['save_model_next_epoch']:
                torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, 'model_epoch%d.pt' % epoch))
                control_dict['save_model_next_epoch'] = False
                with open(self.control_file_path, 'w') as f:
                    yaml.dump(control_dict, f)
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

    def cal_cm_func(self, pred, label, output):
        valid_intervals = output[2]
        valid_pred = [pred[i][valid_intervals[i][0]:valid_intervals[i][1]].numpy() for i in range(len(pred))]
        valid_label = [label[i][valid_intervals[i][0]:valid_intervals[i][1]].numpy() for i in range(len(label))]
        return get_epoch_confusion(np.concatenate(valid_pred).astype(int), np.concatenate(valid_label).astype(int))

    def cal_acc_func(self, pred, label, output):
        valid_intervals = output[2]
        sample_total_snaps = [(valid_intervals[i][1] - valid_intervals[i][0]) for i in range(len(pred))]
        # print(pred)
        # print(label)
        sample_correct_snaps = [len(torch.where(pred[i][valid_intervals[i][0]:valid_intervals[i][1]] ==
                                                label[i][valid_intervals[i][0]:valid_intervals[i][1]])[0])
                                for i in range(len(pred))]
        # sample_correct_snaps = [0 for _ in range(len(pred))]
        # for sample_idx, sample_label in enumerate(label):
        #     sample_pred = pred[sample_idx]
        #     sample_valid_interval = valid_intervals[sample_idx]
        #     for snap_idx in range(sample_valid_interval[0], sample_valid_interval[1]):
        #         if sample_pred[snap_idx] == sample_label[snap_idx]:
        #             sample_correct_snaps[sample_idx] += 1
        # return [sample_correct_snaps[i] / sample_total_snaps[i] for i in range(len(pred))]
        return sum(sample_correct_snaps) / sum(sample_total_snaps)

    def save_epoch_output(self, out_dict, epoch, is_test=True):
        for k, v in out_dict.items():
            file_name = 'test' + k + '_epoch%d.pkl' if is_test else 'train' + k + '_epoch%d.pkl'
            with open(os.path.join(self.log_dir, file_name % epoch), 'wb') as f:
                pickle.dump(v, f)

    def output_collate_fn(self, epoch_output_list):
        batch_out_item_num = len(epoch_output_list[0])
        collated = [[] for _ in range(batch_out_item_num)]
        for batch_idx, batch in enumerate(epoch_output_list):
            for item_idx in range(batch_out_item_num):
                collated[item_idx].extend(batch[item_idx])
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
        plt.plot(self.train_accuracy_list)
        plt.show()
        plt.plot(self.train_loss_list)
        plt.show()
        plt.plot(self.eval_accuracy_list)
        plt.show()
        plt.plot(self.eva_loss_list)
        plt.show()
        plt.plot(self.learning_rate_list)
        plt.show()


def dynamic_import(path):
    components = path.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)
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


def get_epoch_confusion(pred, true):
    assert len(pred) == len(true)
    max_label = np.max(true) + 1
    epoch_confusion_matrix = np.zeros([max_label, max_label])
    for i in range(len(pred)):
        epoch_confusion_matrix[true[i], pred[i]] += 1
    return epoch_confusion_matrix


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


if __name__ == '__main__':
    setting_name = 'cnnv1'
    for lr in [0.1]:
        print('init lr %s' % str(lr))
        for fold in range(1, 6):
            config_path = './resources/config/%s.yaml' % setting_name
            model_arg = {'model_type': 'net.cnnv1.CNNModel'}
            optimizer_arg = {'optimizer_type': 'SGD', 'lr': lr}
            scheduler_arg = {'scheduler_type': 'StepLR', 'step_size': 64, 'gamma': 0.1}
            data_arg = {'dataset': 'dataset.cnnv1dataset.CNNv1Dataset',
                        'train_dataset_arg': {'index_file_path': r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\fold\div2n_sninhtp\train%d.pkl' % fold},
                        'test_dataset_arg': {'index_file_path': r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\fold\div2n_sninhtp\test%d.pkl' % fold},
                        'batch_size': 8,
                        'shuffle': True,
                        'num_workers': 1,
                        'drop_last': False}
            loss_arg = {'loss_type': 'loss.cnnv1loss.CNNv1Loss'}
            pred_arg = {'pred_type': 'pred.cnnv1pred.CNNv1Pred'}
            output_arg = {'log_dir': './resources/result/%s/%s' % (setting_name, str(lr)),
                          'model_save_dir': './resources/result/%s/%s' % (setting_name, str(lr))}
            with open(config_path, 'w') as f:
                yaml.dump({'model_arg': model_arg, 'optimizer_arg': optimizer_arg, 'scheduler_arg': scheduler_arg,
                           'data_arg': data_arg, 'loss_arg': loss_arg, 'pred_arg': pred_arg, 'output_arg': output_arg,
                           'output_device': 0, 'collate_fn': 'dataset.collate_fn.non_array_to_list'}, f)
            with open(config_path, 'r') as f:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            # model_arg, optimizer_arg, scheduler_arg, data_arg, loss_arg, pred_arg, output_arg = config_dict['model_arg'], config_dict[
            #     'optimizer_arg'], config_dict['scheduler_arg'], config_dict['data_arg'], config_dict['loss_arg'], config_dict['pred_arg'], config_dict[
            #                                                                               'output_arg']
            # train = Train(model_arg, optimizer_arg, scheduler_arg, data_arg, loss_arg, pred_arg, output_arg)
            print('Fold %d' % fold)
            train = Train(**config_dict)
            num_epoch = 256
            eval_step = 1
            # model_save_path = './result'
            train.train(num_epoch, eval_step)
            # train.run_train(num_epoch, eval_step)
