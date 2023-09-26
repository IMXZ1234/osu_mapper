import torch

from system.base_sys import Train


class TrainWord2VecSkipGram(Train):
    def __init__(self, config_dict, task_type, **kwargs):
        super(TrainWord2VecSkipGram, self).__init__(config_dict, task_type, **kwargs)

    def run_train(self):
        self.init_train_state()
        for iter_idx in range(self.epoch):
            # center: [N], data: [N, I]
            for center, data in self.train_iter:
                center = center.to(self.output_device)
                data = data.to(self.output_device)

                batch_size, num_instance = data.shape
                # N -> N, C -> N, I, C -> N*I, C
                center_embedding = self.model(center).unsqueeze(dim=1).expand([-1, num_instance, -1]).reshape([batch_size * num_instance, -1])
                pred = self.model(data.reshape(-1)).reshape([batch_size * num_instance, -1])
                similarity = torch.cosine_similarity(pred, center_embedding)
                # train skipgram using negative sampling
                # each iteration, dataloader yields a single positive and a few negative samples
                label = torch.zeros([batch_size, num_instance], dtype=torch.long, device=data.device)
                label[:, 0] = 1
                label = label.reshape([batch_size * num_instance, -1])
                loss = self.loss(similarity, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

        self.save_model()
