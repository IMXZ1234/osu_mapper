import torch
import yaml
from torch.autograd import Variable

from util import audio_util
from dataset import collate_fn
from util.general_util import dynamic_import


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


class Inference:
    def __init__(self, config_path, model_path, device='cpu'):
        with open(config_path, 'rt', encoding='utf-8') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.dataset = dynamic_import(config_dict['data_arg']['dataset'])
        self.pred = self.load_pred(**config_dict['pred_arg'])
        self.model = self.load_model(**config_dict['model_arg'])
        model_state_dict = torch.load(model_path)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

        if device == 'cpu':
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_pred(self, pred_type, **kwargs):
        pred = dynamic_import(pred_type)(**kwargs)
        return pred

    def load_model(self, model_type, **kwargs):
        net = dynamic_import(model_type)(**kwargs)
        return net

    def run_inference(self, audio_file_path, speed_stars, bpm, start_time, end_time, feature_frames_per_beat=512, snap_divisor=8):
        """
        start_time, end_time should be in microseconds
        """
        audio_data, sample_rate = audio_util.audioread_get_audio_data(audio_file_path)
        data = self.dataset.preprocess(audio_data, sample_rate, speed_stars, bpm, start_time, end_time,
                                       beat_feature_frames=feature_frames_per_beat, snap_divisor=snap_divisor)
        print(data)
        if self.device != torch.device('cpu'):
            self.model.cuda(self.device)
            data = recursive_wrap_data(data, self.device)
        # into batch of size 1 to fit the model input format
        data = [[data[i]] for i in range(len(data))]
        with torch.no_grad():
            output = self.model(data)
        label = self.pred(output)
        print(label)
        return label[0].cpu().numpy()

