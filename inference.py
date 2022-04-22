import torch

from util.general_util import dynamic_import, recursive_wrap_data, recursive_to_cpu


class Inference:
    def __init__(self, pred_arg, model_arg, weights_path, data_arg=None, **kwargs):
        # since Inference object may be initialized before dataset,
        # we only keep a copy of data_arg, and postpone initialization of dataloader
        self.data_arg = data_arg
        self.data_iter = None
        self.pred = self.load_pred(**pred_arg)
        self.model = self.load_model(**model_arg)
        self.weights_path = weights_path
        model_state_dict = torch.load(self.weights_path)
        self.model.load_state_dict(model_state_dict)
        if 'collate_fn' in kwargs:
            self.collate_fn = dynamic_import(kwargs['collate_fn'])
        else:
            self.collate_fn = None
        if 'output_collate_fn' in kwargs:
            self.output_collate_fn = dynamic_import(kwargs['output_collate_fn'])
        else:
            self.output_collate_fn = dynamic_import('dataset.collate_fun.output_collate_fn')
        if 'output_device' in kwargs:
            self.output_device = kwargs['output_device']
        else:
            self.output_device = 'cpu'
        if isinstance(self.output_device, int):
            self.model.cuda(self.output_device)
        self.model.eval()

    def load_data(self,
                  dataset,
                  test_dataset_arg,
                  batch_size=8,
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
        return label

