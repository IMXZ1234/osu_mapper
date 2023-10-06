import collections
import os
import pickle
from tqdm import tqdm

import torch

from system.base_sys import Train


class TrainWord2VecSkipGram(Train):
    def __init__(self, config_dict, task_type='skipgram', **kwargs):
        super(TrainWord2VecSkipGram, self).__init__(config_dict, task_type, **kwargs)
        self.model_save_step_per_epoch = self.config_dict['output_arg'].get('model_save_step_per_epoch', None)

    def run_train(self):
        self.init_train_state()
        if self.model_save_step_per_epoch is not None:
            self.save_model_batch_idx = list(range(0, len(self.train_iter), round(len(self.train_iter) / self.model_save_step_per_epoch)))
        else:
            self.save_model_batch_idx = []
        bar = tqdm(range(self.epoch))
        for current_epoch in bar:
            # center: [N], data: [N, I], label: [N, I]
            for batch_idx, (center, data, label, mask) in enumerate(self.train_iter):
                center = center.long().to(self.output_device)
                data = data.long().to(self.output_device)
                label = label.float().to(self.output_device)
                mask = mask.long().to(self.output_device)

                batch_size, num_instance = data.shape
                # N -> N, C -> N, I, C -> N*I, C
                # center_embedding = self.model.forward(center).unsqueeze(dim=1).expand([-1, num_instance, -1]).reshape([batch_size * num_instance, -1])
                # context_embedding = self.model.forward_context(data.reshape(-1)).reshape([batch_size * num_instance, -1])
                # similarity = torch.cosine_similarity(context_embedding, center_embedding)
                # N -> N, C -> N, C, I
                center_embedding = self.model.forward(center).unsqueeze(dim=-1)
                # N, I -> N*I -> N*I, C -> N, I, C
                context_embedding = self.model.forward_context(data.reshape(-1)).reshape([batch_size, num_instance, -1])
                # N, I, 1 -> N, I
                similarity = torch.bmm(context_embedding, center_embedding)
                # # train skipgram using negative sampling
                # # each iteration, dataloader yields a single positive and a few negative samples
                # label = torch.zeros([batch_size, num_instance], dtype=torch.long, device=data.device)
                # label[:, 0] = 1
                # label = label.reshape([batch_size * num_instance, -1])
                # print(similarity)
                # print(similarity.shape)
                # print(label)
                # print(label.shape)
                # print(mask)
                # print(mask.shape)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(similarity.reshape(-1), label.reshape(-1), weight=mask.reshape(-1))
                # loss = self.loss(similarity.reshape(-1), label.reshape(-1), weight=mask.reshape(-1))

                bar.set_postfix(collections.OrderedDict(
                    {
                        'loss': loss.item()
                    }
                ))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_size in self.save_model_batch_idx:
                    self.save_embedding(current_epoch, batch_idx)

            self.scheduler.step()
            if current_epoch % self.model_save_step == 0:
                self.save_model(current_epoch)
                self.save_embedding(current_epoch, -1)

        self.save_model()
        self.save_embedding()

    def save_embedding(self, epoch=-1, batch=-1):
        print('saved embedding %d %d' % (epoch, batch))
        with open(os.path.join(self.model_save_dir, 'embedding_center%d_%d.pkl' % (epoch, batch)), 'wb') as f:
            pickle.dump(self.model.embedding_center.data.cpu().numpy(), f)
        with open(os.path.join(self.model_save_dir, 'embedding_context%d_%d.pkl' % (epoch, batch)), 'wb') as f:
            pickle.dump(self.model.embedding_context.data.cpu().numpy(), f)
