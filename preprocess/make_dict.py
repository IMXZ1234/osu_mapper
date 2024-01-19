import os
import pickle
from tqdm import tqdm
import numpy as np

"""
# x_pos_seq = (x_pos_seq + 180) / (691 + 180)
# y_pos_seq = (y_pos_seq + 82) / (407 + 82)
x_pos_seq = (x_pos_seq - 256) / 512
y_pos_seq = (y_pos_seq - 192) / 384
label = np.stack([snap_type, x_pos_seq, y_pos_seq], axis=1)
"""
if __name__ == '__main__':
    pass
    # # make label dict
    # root_dir = r'/home/xiezheng/data/preprocessed_v6/label'
    # for filename in tqdm(os.listdir(root_dir)):
    #     filepath = os.path.join(root_dir, filename)
    #     with open(filepath, 'rb') as f:
    #         label = pickle.load(f)
    #     label = np.stack([t, x, y], axis=1)
    #     with open(filepath, 'wb') as f:
    #         pickle.dump(label, f)
