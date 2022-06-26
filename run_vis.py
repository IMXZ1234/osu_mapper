from vis import vis_model

import torch
import pickle
import numpy as np


if __name__ == '__main__':
    # with open(r'C:\Users\asus\coding\python\osu_mapper\resources\result\cganv1_sample_beats_48\0.1\1\trainepoch_gen_output_list_epoch118.pkl', 'rb') as f:
    #     gen_out = pickle.load(f)
    # print(torch.argmax(gen_out[12], dim=1))
    # print(len(gen_out[0]))
    with open(
r'C:\Users\asus\coding\python\osu_mapper\resources\data\fit\rnnv3_nolabel\train1_label.pkl',
            'rb') as f:
        labels = pickle.load(f)
    for sample_label in labels:
        sample_label = sample_label[:(len(sample_label)//4)*4].reshape([-1, 4])
        for k in sample_label:
            if np.equal(np.array([2, 0, 2, 0]), sample_label[k]).all():
                print(sample_label)
                print('at %d' % k * 4)
    # print(torch.argmax(gen_out[12], dim=1))
    # print(len(gen_out[0]))
    # mv = vis_model.ModelParamViewer(
    #     r'resources/config/inference/rnnv1_lr0.01.yaml',
    # )
