import logging
import os

import numpy as np
import tensorboardX
import torch
from tqdm import tqdm

from system.base_sys import Train
from util.general_util import dynamic_import, recursive_wrap_data
from util.plt_util import plot_signal

logging.getLogger('matplotlib').setLevel(logging.WARNING)


class TrainRegression(Train):
    def __init__(self, config_dict, train_type='classification', **kwargs):
        super(TrainRegression, self).__init__(config_dict, train_type, **kwargs)

        if 'embedding_decoder' in self.config_dict:
            self.embedding_output_decoder = dynamic_import(self.config_dict['embedding_decoder'])(**self.config_dict['embedding_decoder_args'])
        self.lambda_cls = self.config_dict['train_arg']['lambda_cls']

        log_dir = self.config_dict['output_arg']['log_dir']
        self.tensorboard_writer = tensorboardX.SummaryWriter(log_dir)

    def run_train(self):
        self.init_train_state()
        self.noise = torch.randn([1, 128], device=self.output_device)
        for epoch in range(self.start_epoch, self.epoch):
            if isinstance(self.model, (list, tuple)):
                for model in self.model:
                    model.train()
            else:
                self.model.train()
            self.train_epoch(epoch)

    def train_epoch(self, epoch):
        epoch_loss = []
        for batch, (cond_data, real_gen_output, cls_label) in enumerate(tqdm(self.train_iter)):
            self.optimizer.zero_grad()
            cond_data = recursive_wrap_data(cond_data, self.output_device)
            real_gen_output = recursive_wrap_data(real_gen_output, self.output_device)
            cls_label = recursive_wrap_data(cls_label, self.output_device)
            coord_output, embed_output = self.model((cond_data, cls_label), self.noise)
            loss = self.loss(coord_output, real_gen_output[0]) + self.loss(embed_output, real_gen_output[1])
            loss.backward()
            epoch_loss.append(loss.item())
            # if self.grad_alter_fn is not None:
            #     self.grad_alter_fn(self.model, **self.grad_alter_fn_arg)
            self.optimizer.step()

        self.scheduler.step()

        print('generating...')
        self.log_gen_output_embedding((coord_output, embed_output), epoch, 'gen', 1)
        self.log_gen_output_embedding(real_gen_output, epoch, 'label', 1)

        print('mean loss %.4f' % (sum(epoch_loss) / len(epoch_loss)))

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
