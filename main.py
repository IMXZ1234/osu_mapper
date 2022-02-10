from __future__ import print_function
import argparse
import os
import time
import yaml
import pickle
import shutil
import random
import inspect
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

warnings.filterwarnings("ignore")


def seed_torch(seed):
    random.seed(seed)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_parser():  # 参数
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Framework')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument(
        '--model-saved-name',
        default='./work_dir/temp',
        help='directory for saved model parameters')
    parser.add_argument(
        '--feature_save_dir',
        default='./work_dir/temp',
        help='directory for saved features, used to view training details')
    parser.add_argument(
        '--config',
        default='./config/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=True,
        help='if ture, the classification score will be stored')
    parser.add_argument(
        '--save-sample-order',
        type=str2bool,
        default=True,
        help='if ture, the sample name in each training batch will be logged')
    parser.add_argument(
        '--save-feature',
        type=str2bool,
        default=True,
        help='if ture, the feature output by models will be saved')
    parser.add_argument(
        '--save-feature-batch-index',
        type=list,
        default=None,
        help='batches in which feature will be saved. if None, save features in all batches.')
    parser.add_argument(
        '--save-training-state',
        type=str2bool,
        default=False,
        help='if ture, the training state will be saved,'
             ' for continuation after unexpected termination of training process')
    parser.add_argument(
        '--continue-training',
        type=str2bool,
        default=False,
        help='if ture, use the saved training state to continue training')

    # visualize and debug
    parser.add_argument(
        '--save-interval',
        type=int,
        default=5,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-model',
        type=str2bool,
        default=False,
        help='save model or not')
    # parser.add_argument(
    #     '--model_save_dir',
    #     frame_type=print_item,
    #     default='./work_dir/temp',
    #     help='target_dir to save model')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder.Feeder', help='data loader will be used')
    parser.add_argument(
        '--collate_fn', type=str, default=None, help='user defined collate function')
    parser.add_argument(
        '--train_sampler', default=None, help='training sampler for the training loader')
    parser.add_argument(
        '--train-sampler-args',
        default=dict(),
        help='the arguments of training sampler')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=0,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--num_class', default=5, help='num of final classification classes')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--scheduler',
        default=None,
        help='the scheduler frame_type')
    parser.add_argument(
        '--scheduler-args',
        type=dict,
        default=dict(),
        help='the arguments of scheduler')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--batch-train-func',
        default='main.train_batch_simple',
        help='the weights for network initialization')
    parser.add_argument(
        '--segmentation-net-weights',
        default=None,
        help='the weights for segmentation network')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # loss
    # parser.add_argument(
    #     '--loss-func', type=str, choices=['cross_entropy', 'mse', 'minor_diff_lenient'], default='cross_entropy', help='loss function to use')
    parser.add_argument(
        '--loss-func', type=str, default='cross_entropy', help='loss function to use')
    parser.add_argument(
        '--loss-func-args', type=dict, default=dict(), help='loss function arguments')

    # predict function
    parser.add_argument(
        '--pred-func', type=str, default='pred_func.default_pred.Default_Pred',
        help='prediction function to use')
    parser.add_argument(
        '--pred-func-args', type=dict, default=dict(), help='prediction function arguments')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='frame_type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=8, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=8, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    return parser


def train_batch_simple(data, model):
    output = model(data)
    return output, None


def train_batch_with_output_feature(data, model):
    output, feature = model(data)
    return output, feature


class Processor:
    """
    Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        print('self.arg.eval_interval: ', self.arg.eval_interval)
        self.save_arg()
        if arg.phase == 'train':
            train_writer_dir = os.path.join(arg.model_saved_name, 'train')
            if not os.path.exists(train_writer_dir):
                os.makedirs(train_writer_dir)
            eval_writer_dir = os.path.join(arg.model_saved_name, 'eval')
            if not os.path.exists(eval_writer_dir):
                os.makedirs(eval_writer_dir)
            self.train_writer = SummaryWriter(train_writer_dir, 'train')
            self.eval_writer = SummaryWriter(eval_writer_dir, 'eval')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_scheduler()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

        self.feature_save_dir = self.arg.feature_save_dir
        if self.arg.save_sample_order:
            self.sample_order_dir = os.path.join(self.arg.work_dir, 'sample_order')
            if not os.path.exists(self.sample_order_dir):
                os.makedirs(self.sample_order_dir)

        self.training_state_save_file = os.path.join(self.arg.work_dir, 'training_state')
        self.training_state_save_file_temp = os.path.join(self.arg.work_dir, 'training_state_temp')


    def load_scheduler(self):
        if self.arg.scheduler == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.arg.scheduler_args['step_size'], gamma=self.arg.scheduler_args['gamma'])
        else:
            raise ValueError()

    def load_data(self):  # 加载数据
        Feeder = dynamic_import(self.arg.feeder)
        # DataLoader分train和test存在字典中
        self.data_loader = dict()

        print('self.arg.collate_fn')
        print(self.arg.collate_fn)
        if self.arg.collate_fn is not None:
            collate_fn = dynamic_import(self.arg.collate_fn)
        else:
            collate_fn = None

        if self.arg.phase == 'train':
            train_dataset = Feeder(**self.arg.train_feeder_args)
            train_dataloader_param_dict = {
                'dataset':train_dataset,
                'batch_size':self.arg.batch_size,
                'shuffle':False,
                'num_workers':self.arg.num_worker,
                'drop_last':False}
            if collate_fn is not None:
                train_dataloader_param_dict['collate_fn'] = collate_fn
            if self.arg.train_sampler != None:
                train_dataloader_param_dict['sampler'] = dynamic_import(self.arg.train_sampler)(data_source=train_dataset, **self.arg.train_sampler_args)
            self.data_loader['train'] = torch.utils.data.DataLoader(**train_dataloader_param_dict)
        test_dataloader_param_dict = {
            'dataset':Feeder(**self.arg.test_feeder_args),
            'batch_size':self.arg.test_batch_size,
            'shuffle':False,
            'num_workers':self.arg.num_worker,
            'drop_last':False}
        if collate_fn is not None:
            test_dataloader_param_dict['collate_fn'] = collate_fn
        self.data_loader['test'] = torch.utils.data.DataLoader(**test_dataloader_param_dict)

    def load_model(self):  # 加载模型
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = dynamic_import(self.arg.model)
        # 把模型所在的源文件存到work_dir
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        if self.arg.model == 'net.integrated.Model' or self.arg.model == 'net.integrated_agg.Model':
            self.model.load_segmentation_net_weights(self.arg.segmentation_net_weights)
        if self.arg.loss_func == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss().cuda(output_device)  # 定义损失函数
        elif self.arg.loss_func == 'mse':
            self.loss = nn.MSELoss().cuda(output_device)
        else:
            # self.loss = dynamic_import('loss_func.minor_diff_lenient.Minor_Diff_Lenient')(**self.arg.loss_func_args).cuda(output_device)
            self.loss = dynamic_import(self.arg.loss_func)(**self.arg.loss_func_args).cuda(output_device)

        self.pred_func = dynamic_import(self.arg.pred_func)(**self.arg.pred_func_args).cuda(output_device)

        if self.arg.weights:
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                # Model's state dict contains keys not included in weights.
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                # Only updates Model's state dict items whose keys are included in weights.
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:  # 在多个gpu上运行
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

        self.batch_train_func = dynamic_import(self.arg.batch_train_func)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def save_arg(self):
        # 在工作目录保存一份运行时的参数配置文件
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def get_epoch_confusion(self, pred, true):
        assert len(pred) == len(true)
        max_label = self.arg.num_class
        epoch_confusion_matrix = np.zeros([max_label, max_label])
        for i in range(len(pred)):
            epoch_confusion_matrix[true[i], pred[i]] += 1
        return epoch_confusion_matrix

    def top_k(self, score, true, k):
        # print('score\n',score)
        # print('true\n',true)
        rank = score.argsort()
        # print(rank)
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(true)]
        # print(hit_top_k)
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def print_log(self, print_item, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            print_item = "[ " + localtime + ' ] ' + str(print_item)
        print(print_item)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(print_item, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_features_old(self, features, is_test, epoch, batch_idx):
        for (feature_name, value) in features.items():
            if is_test:
                feature_save_path = os.path.join(self.feature_save_dir, feature_name, 'test')
            else:
                feature_save_path = os.path.join(self.feature_save_dir, feature_name, 'train')
            if not os.path.exists(feature_save_path):
                os.makedirs(feature_save_path)
            with open(os.path.join(feature_save_path, 'epoch%dbatch%d' % (epoch, batch_idx)), 'wb') as f:
                if isinstance(value, torch.Tensor):
                    # most features generated by network is gpu tensor
                    value = value.cpu().detach().numpy()
                pickle.dump(value, f)

    def save_features(self, features, is_test, loader_name, epoch):
        for (feature_name, value) in features.items():
            if is_test:
                new_feature_save_root_dir = os.path.join(self.feature_save_dir, feature_name, 'new', 'test')
            else:
                new_feature_save_root_dir = os.path.join(self.feature_save_dir, feature_name, 'new', 'train')
            if loader_name == 'test':
                new_feature_save_dir = os.path.join(new_feature_save_root_dir, 'loader_test')
            else:
                new_feature_save_dir = os.path.join(new_feature_save_root_dir, 'loader_train')
            new_feature_save_file = os.path.join(new_feature_save_dir, 'epoch%d' % epoch)
            if not os.path.exists(new_feature_save_dir):
                os.makedirs(new_feature_save_dir)
            if not os.path.exists(new_feature_save_file):
                with open(new_feature_save_file, 'wb') as f:
                    pickle.dump([], f)
            with open(new_feature_save_file, 'rb') as f:
                all_value = pickle.load(f)
            with open(new_feature_save_file, 'wb') as f:
                # print(value)
                if isinstance(value, torch.Tensor):
                    # most features generated by network is gpu tensor
                    # print('value is tensor')
                    value = value.cpu().detach().numpy()
                elif isinstance(value, (list, tuple)):
                    # if is list of gpu tensor
                    # print('value is list')
                    if len(value) != 0:
                        if isinstance(value[0], torch.Tensor):
                            value = [v.cpu().detach().numpy() for v in value]
                #     print('value', value)
                # print('all_value', all_value)
                all_value.append(value)
                print(len(all_value))
                pickle.dump(all_value, f)

    def feature_dict_to_cpu(self, features):
        for (feature_name, value) in features.items():
            features[feature_name] = recursive_to_cpu(value)
        return features

    def save_epoch_feature(self, epoch_feature_list, is_test, loader_name, epoch):
        for feature_dict in epoch_feature_list.values():
            feature_name_list = list(feature_dict.keys())
            break
        # print(feature_name_list)
        for feature_name in feature_name_list:
            if is_test:
                feature_save_root_dir = os.path.join(self.feature_save_dir, feature_name, 'test')
            else:
                feature_save_root_dir = os.path.join(self.feature_save_dir, feature_name, 'train')
            if loader_name == 'test':
                feature_save_dir = os.path.join(feature_save_root_dir, 'loader_test')
            else:
                feature_save_dir = os.path.join(feature_save_root_dir, 'loader_train')
            feature_save_file = os.path.join(feature_save_dir, 'epoch%d' % epoch)
            if not os.path.exists(feature_save_dir):
                os.makedirs(feature_save_dir)
            with open(feature_save_file, 'wb') as f:
                all_value = {}
                for batch_index, batch_feature_dict in epoch_feature_list.items():
                    value = batch_feature_dict[feature_name]
                    all_value[batch_index] = value
                pickle.dump(all_value, f)


    def recursive_wrap_data(self, data):
        """
        recursively wrap tensors in data into Variable

        :param data:
        :return:
        """
        if isinstance(data, list):
            for i in range(len(data)):
                data[i] = self.recursive_wrap_data(data[i])
        elif isinstance(data, torch.Tensor):
            return Variable(data.float().cuda(self.output_device), requires_grad=False)
        return data

    def train_epoch(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = loader
        sample_order_list = []

        epoch_feature_dict = {}
        for batch_idx, (data, label, index) in enumerate(process):
            # print('train epoch')
            # print('batch_idx')
            # print(batch_idx)
            # print('index')
            # print(index)
            self.global_step += 1
            # 加载数据和标签
            data = self.recursive_wrap_data(data)
            # if isinstance(data, list):
            #     for i in range(len(data)):
            #         if isinstance(data[i], torch.Tensor):
            #             data[i] = Variable(data[i].float().cuda(self.output_device), requires_grad=False)
            # else:
            #     data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()
            if self.arg.save_sample_order:
                sample_order_list.extend(index)
            output, features = self.model(data)
            # print('in main')
            # print(type(output[1][0][0]))
            # print(output[1][0][0])
            # print('in iterator')
            # print(features)
            if features is not None:
                if self.arg.save_feature_batch_index is None:
                    # move to cpu as soon as possible to save gpu memory
                    epoch_feature_dict[batch_idx] = self.feature_dict_to_cpu(features)
                else:
                    if (batch_idx in self.arg.save_feature_batch_index) or \
                        (batch_idx == len(process) - 1 and -1 in self.arg.save_feature_batch_index):
                        epoch_feature_dict[batch_idx] = self.feature_dict_to_cpu(features)

            # print('before loss')
            # print(type(output[1][0][0]))
            # print(output[1][0][0])
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.item())
            timer['model'] += self.split_time()

            predict_label = self.pred_func(output)

            acc = torch.mean((predict_label == label.data).float())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']

            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        if self.arg.save_feature and len(epoch_feature_dict) != 0:
            print('saving train features...')
            self.save_epoch_feature(epoch_feature_dict, False, 'train', epoch)

        self.scheduler.step()
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log('\tlearning rate: {:.5f}.'.format(self.lr))
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))

        # save model using torch.save
        if save_model:
            state_dict = self.model.state_dict()
            torch.save(state_dict, self.arg.model_saved_name + '/' + "epoch-" + str(epoch+1) + '.pt')

        if self.arg.save_sample_order:
            with open(os.path.join(self.sample_order_dir, 'train_epoch%d' % epoch), 'wb') as f:
                pickle.dump(sample_order_list, f)
        # save training state
        # use temp file and renaming strategy to avoid broken model,
        # which will happen if termination happens in torch.save() call
        if self.arg.save_training_state:
            state_dict = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
            torch.save(state_dict, self.training_state_save_file_temp)
            if os.path.exists(self.training_state_save_file):
                os.remove(self.training_state_save_file)
            if os.path.exists(self.training_state_save_file_temp):
                os.rename(self.training_state_save_file_temp, self.training_state_save_file)

    # 记录一个epoch结束后的损失函数值、准确率（前k（k=1）中包含真实标签），所有样本的预测结果保存到result_file，预测错误的样本保存到wrong_file
    def eval(self, epoch, save_score=False, loader_name='test', wrong_file=None, result_file=None):
        self.model.eval_epoch()
        if wrong_file is not None:
            f_w = open(wrong_file, 'a')
            f_w.write('Eval epoch: {}\n'.format(epoch + 1))
        if result_file is not None:
            f_r = open(result_file, 'a')
            f_r.write('Eval epoch: {}\n'.format(epoch + 1))
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        print('loader_name\n', loader_name)
        loss_value = []
        score_frag = []
        predict_frag = []
        true_frag = []
        process = self.data_loader[loader_name]
        batch_num = 0
        epoch_feature_dict = {}
        for batch_idx, (data, label, index) in enumerate(process):
            # print('batch_idx')
            # print(batch_idx)
            # print('data')
            # print(data)
            # print('label')
            # print(label)
            # print('index')
            # print(index)
            data = self.recursive_wrap_data(data)
            # if isinstance(data, list):
            #     for i in range(len(data)):
            #         if isinstance(data[i], torch.Tensor):
            #             data[i] = Variable(data[i].float().cuda(self.output_device), requires_grad=False)
            # else:
            #     data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)

            output, features = self.model(data)
            if features is not None:
                if self.arg.save_feature_batch_index is None:
                    epoch_feature_dict[batch_idx] = self.feature_dict_to_cpu(features)
                else:
                    if (batch_idx in self.arg.save_feature_batch_index) or \
                            (batch_idx == len(process) - 1 and -1 in self.arg.save_feature_batch_index):
                        epoch_feature_dict[batch_idx] = self.feature_dict_to_cpu(features)
            loss = self.loss(output, label)

            predict_label = self.pred_func(output)

            # If output is a list, we assume that the first item is always data tensor.
            if isinstance(output, list):
                score_frag.append(output[0].data.cpu().numpy())
            else:
                score_frag.append(output.data.cpu().numpy())
            loss_value.append(loss.item())
            # 一个batch的预测标签
            batch_predict = list(predict_label.cpu().numpy().copy())
            batch_true = list(label.data.cpu().numpy().copy())
            # self.print_log('batch_predict')
            # self.print_log(batch_predict)
            # self.print_log('batch_true')
            # self.print_log(batch_true)
            predict_frag.append(batch_predict)
            true_frag.append(batch_true)

            if wrong_file is not None or result_file is not None:
                # i:这个batch内该样本的序号
                for i, x in enumerate(batch_predict):
                    if result_file is not None:
                        f_r.write(str(index[i]) + ',' + str(x) + ',' + str(batch_true[i]) + '\n')
                    if x != batch_true[i] and wrong_file is not None:
                        # index:整个数据集内该样本的序号
                        f_w.write(str(index[i]) + ',' + str(x) + ',' + str(batch_true[i]) + '\n')
            batch_num += 1

        if self.arg.save_feature and len(epoch_feature_dict) != 0:
            print('saving test features...')
            self.save_epoch_feature(epoch_feature_dict, True, loader_name, epoch)

        # 把所有batch的打分结果拼起来
        # print('\n\n\n')
        # print('score_frag')
        # print(score_frag)
        # print('\n\n\n')
        score = np.concatenate(score_frag)
        loss = np.mean(loss_value)
        predict = np.concatenate(predict_frag)
        true = np.concatenate(true_frag)
        # print confusion matrix for this evaluation epoch
        self.print_log(str(self.get_epoch_confusion(predict, true)), print_time=False)
        # 前k大评分对应label包含真实标签的准确率
        # accuracy = self.data_loader[ln].dataset.top_k(score, 1)
        accuracy = self.top_k(score, true, 1)
        if accuracy > self.best_acc:
            self.best_acc = accuracy
        print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
        if self.arg.phase == 'train':
            self.eval_writer.add_scalar('loss', loss, self.global_step)
            self.eval_writer.add_scalar('acc', accuracy, self.global_step)

        score_dict = dict(
            zip(self.data_loader[loader_name].dataset.sample_name, score))
        self.print_log('\tMean {} loss of {} batches: {}.'.format(
            loader_name, len(self.data_loader[loader_name]), np.mean(loss_value)))
        for k in self.arg.show_topk:
            self.print_log('\tTop{}: {:.2f}%'.format(
                k, 100 * self.top_k(score, true, k)))

        if save_score:  # 保存全连接层输出的分数
            with open('{}/epoch{}_{}_score.pkl'.format(
                    self.arg.work_dir, epoch + 1, loader_name), 'wb') as f:
                pickle.dump(score_dict, f)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            if self.arg.continue_training:
                if os.path.exists(self.training_state_save_file):
                    training_state = torch.load(self.training_state_save_file)
                elif os.path.exists(self.training_state_save_file_temp):
                    training_state = torch.load(self.training_state_save_file_temp)
                else:
                    training_state = None
                    print('training state not found, train from the beginning')
                if training_state is not None:
                    self.model.load_state_dict(training_state['model'])
                    self.optimizer.load_state_dict(training_state['optimizer'])
                    self.arg.start_epoch = training_state['epoch']
                    print('loaded training state, start training from epoch %d' % self.arg.start_epoch)

            # 步长
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # 每隔save_interval存储一次模型
                save_model = self.arg.save_model and (((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch))
                self.train_epoch(epoch, save_model=save_model)
                # evaluate every eval_interval, using test data and train data
                # print('self.arg.eval_interval: ', self.arg.eval_interval)
                # print('%', (epoch + 1) % self.arg.eval_interval)
                if ((epoch + 1) % self.arg.eval_interval == 0) or (epoch + 1 == self.arg.num_epoch):
                    self.eval(epoch, save_score=self.arg.save_score, loader_name='test',
                              wrong_file=self.arg.work_dir + '_test_wrong.txt',
                              result_file=self.arg.work_dir + '_test_result.txt')
                    self.eval(epoch, save_score=self.arg.save_score, loader_name='train',
                              wrong_file=self.arg.work_dir + '_train_wrong.txt',
                              result_file=self.arg.work_dir + '_train_result.txt')
            print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)
            # delete saved training state when training is over
            if os.path.exists(self.training_state_save_file):
                os.remove(self.training_state_save_file)
            if os.path.exists(self.training_state_save_file_temp):
                os.remove(self.training_state_save_file_temp)

        elif self.arg.phase == 'test':
            # if not self.arg.test_feeder_args['debug']:
            #     wf = self.arg.model_saved_name + '_wrong.txt'
            #     rf = self.arg.model_saved_name + '_right.txt'
            # else:
            #     wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name='test',
                      wrong_file=self.arg.model_saved_name + '_test_wrong.txt',
                        result_file=self.arg.model_saved_name + '_test_result.txt')
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name='train',
                      wrong_file=self.arg.model_saved_name + '_train_wrong.txt',
                      result_file=self.arg.model_saved_name + '_train_result.txt')
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'bbox_cfd_thr_0_original', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def dynamic_import(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def recursive_to_cpu(value):
    """
    Shallow copy here for lists.

    :param value:
    :return:
    """
    if isinstance(value, torch.Tensor):
        # most features generated by network is gpu tensor
        # print('value is tensor')
        return value.cpu().detach().numpy()
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


if __name__ == '__main__':
    seed_torch(2)
    parser = get_parser()

    p = parser.parse_args()
    # Has an argument specifying the config file.
    if p.config is not None:
        # Load arg form config file to default_arg.
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        # Go through config file and set as default.
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        # Here defaults are set on interpreter level,
        # and will overwrite default settings defined using add_argument()
        # on parameter level.
        parser.set_defaults(**default_arg)
    # No matter how default value changes, parameters from command line
    # will always override default values.
    arg = parser.parse_args()
    processor = Processor(arg)
    processor.start()
