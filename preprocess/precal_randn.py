import numpy as np
import multiprocessing
import pickle
import os


def cal_randn(size, save_path, seed=None):
    print('in proc %d' % seed)
    rs = np.random.RandomState(seed)
    with open(save_path, 'wb') as f:
        pickle.dump(
            rs.randn(size), f
        )


if __name__ == '__main__':
    save_dir = r'/home/data1/xiezheng/osu_mapper/temp/cal_rand'
    os.makedirs(save_dir, exist_ok=True)
    print('saving to %s' % save_dir)
    rand_size = 32 * 16 * 1024
    save_path = r'/home/data1/xiezheng/osu_mapper/preprocessed_v5/randn_%d' % rand_size
    print('saving result to %s' % save_path)

    pool = multiprocessing.Pool()

    for i in range(64):
        pool.apply_async(
            cal_randn, (rand_size // 64, os.path.join(save_dir, str(i)), i)
        )
    pool.close()
    print('waiting')
    pool.join()
    print('finished')

    all_result = []
    for i in range(64):
        with open(os.path.join(save_dir, str(i)), 'rb') as f:
            all_result.append(pickle.load(f))
    with open(save_path, 'wb') as f:
        pickle.dump(np.concatenate(all_result), f)
