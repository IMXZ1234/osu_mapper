import functools
import random

import torch
import numpy as np
from torch.autograd import Variable

from util.general_util import dynamic_import, recursive_wrap_data, recursive_to_cpu
from nn.dataset import collate_fn


class Inference:
    def __init__(self, pred_arg, model_arg, weights_path, data_arg=None, **kwargs):

        if 'random_seed' in kwargs:
            seed = kwargs['random_seed']
            print('set random seed to %d' % seed)
            random.seed(seed)
            torch.random.manual_seed(seed)
        # since Inference object may be initialized before dataset,
        # we only keep a copy of data_arg, and postpone initialization of dataloader
        self.data_arg = data_arg
        self.data_iter = None
        self.pred = self.load_pred(**pred_arg)
        self.model = self.load_model(**model_arg)
        self.weights_path = weights_path
        model_state_dict = torch.load(self.weights_path)
        # if kwargs['train_type'] == 'gan':
        #     # only load generator
        #     model_state_dict = model_state_dict[0]
        self.model.load_state_dict(model_state_dict)
        if 'collate_fn' in kwargs:
            self.collate_fn = dynamic_import(kwargs['collate_fn'])
        else:
            self.collate_fn = None
        if 'output_collate_fn' in kwargs:
            self.output_collate_fn = dynamic_import(kwargs['output_collate_fn'])
        else:
            self.output_collate_fn = collate_fn.output_collate_fn
        if 'output_device' in kwargs:
            self.output_device = kwargs['output_device']
        else:
            self.output_device = 'cpu'
        if 'train_type' in kwargs:
            if kwargs['train_type'] == 'gan':
                if 'noise_size' in kwargs:
                    self.noise_size = kwargs['noise_size']
        if isinstance(self.output_device, int):
            self.model.cuda(self.output_device)
        if 'model_eval_mode' not in kwargs or kwargs['model_eval_mode']:
            # use eval mode by default
            self.model.eval()
        else:
            print('model use train mode')

    def load_data(self,
                  dataset,
                  test_dataset_arg,
                  batch_size=1,
                  num_workers=1,
                  **kwargs):
        test_iter = torch.utils.data.DataLoader(
            dataset=dynamic_import(dataset)(**test_dataset_arg),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            drop_last=False)
        return test_iter

    def load_pred(self, pred_type, **kwargs):
        if pred_type is None or pred_type == 'argmax':
            pred = functools.partial(torch.argmax, dim=1)
        elif pred_type == 'no_pred':
            pred = lambda x: x
        else:
            pred = dynamic_import(pred_type)(**kwargs)
        return pred

    def load_model(self, model_type, **kwargs):
        net = dynamic_import(model_type)(**kwargs)
        return net

    def set_data_arg(self, data_arg):
        self.data_arg = data_arg

    def run_inference(self):
        """
        Unlike in Train, we initialize dataloader(data_iter) right before passing data through model,
        because same model may be used on different datasets.
        """
        if self.data_arg is None:
            print('data_arg not specified!')
        self.data_iter = self.load_data(**self.data_arg)
        epoch_output_list = []
        for batch, (data, index) in enumerate(self.data_iter):
            data = recursive_wrap_data(data, self.output_device)
            output = self.model(data)
            epoch_output_list.append(recursive_to_cpu(output))
        # print(epoch_output_list)
        epoch_output_list = self.output_collate_fn(epoch_output_list)
        label = self.pred(epoch_output_list)
        return label.numpy()

    def run_inference_gan(self):
        """
        Unlike in Train, we initialize dataloader(data_iter) right before passing data through model,
        because same model may be used on different datasets.
        """
        if self.data_arg is None:
            print('data_arg not specified!')
        self.data_iter = self.load_data(**self.data_arg)
        epoch_output_list = []
        for batch, (data, index) in enumerate(self.data_iter):
            data = recursive_wrap_data(data, self.output_device)
            if self.noise_size is not None:
                noise = Variable(torch.randn(self.noise_size).cuda(self.output_device), requires_grad=False)
                output = self.model(noise, data)[0]
            else:
                output = self.model(data)[0]
            epoch_output_list.append(recursive_to_cpu(output))
        # print(epoch_output_list)
        epoch_output_list = self.output_collate_fn(epoch_output_list)
        label = self.pred(epoch_output_list)
        return label.numpy()

    def run_inference_sample(self, data):
        """
        Unlike in Train, we initialize dataloader(data_iter) right before passing data through model,
        because same model may be used on different datasets.
        """
        data = torch.tensor(data, dtype=torch.float32)
        # to batch of 1
        data = data.reshape([1] + list(data.shape))
        data = recursive_wrap_data(data, self.output_device)
        output = self.model(data)
        label = self.pred(recursive_to_cpu(output))[0]
        print('label')
        print(label)
        return label.numpy()

    def run_inference_model_sample(self):
        """
        Unlike in Train, we initialize dataloader(data_iter) right before passing data through model,
        because same model may be used on different datasets.
        """
        if self.data_arg is None:
            print('data_arg not specified!')
        self.data_iter = self.load_data(**self.data_arg)
        epoch_output_list = []
        for batch, (data, index) in enumerate(self.data_iter):
            data = recursive_wrap_data(data, self.output_device)
            output, h = self.model.sample(data)
            epoch_output_list.append(output.reshape([-1]))
        print(epoch_output_list)
        output = self.data_iter.dataset.cat_sample_labels(epoch_output_list)
        output = recursive_to_cpu(output)
        print('output')
        print(output)
        return [label.numpy() for label in output]

    def run_inference_sample_rnn(self, data, state):
        """
        Unlike in Train, we initialize dataloader(data_iter) right before passing data through model,
        because same model may be used on different datasets.
        """
        # to batch of 1
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32).reshape([1] + list(data.shape))
        else:
            data = [torch.tensor(d, dtype=torch.float32).reshape([1] + list(d.shape))
                    if isinstance(d, np.ndarray) else torch.tensor(d, dtype=torch.float32).reshape([1, 1])
                    for d in data]
        data = recursive_wrap_data(data, self.output_device)
        output, state = self.model(data, state)
        # print('output')
        # print(output)
        # print('state')
        # print(state)
        # [0] get the label for the first batch
        label = self.pred(recursive_to_cpu(output))[0]
        # print('label')
        # print(label)
        return label.numpy(), state

