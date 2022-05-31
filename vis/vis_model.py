from nn import inference
import yaml


class ModelParamViewer:
    def __init__(self, config_path, pt_path=None):
        with open(config_path, 'r') as f:
            config_dict = yaml.load(f, yaml.FullLoader)
        if pt_path is not None:
            config_dict['weights'] = pt_path
        inf = inference.Inference(**config_dict)
        for name, param in inf.model.named_parameters():
            print(name)
            print(param)
            print(param.shape)


