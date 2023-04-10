import os
import pickle

from preprocess.dataset import recur_dataset
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter

from preprocess import db, prepare_data

DEFAULT_MODEL_SAVE_DIR = r'./resources/data/fit/model'
DEFAULT_RANDOM_SEED = 404


def count_labels(labels):
    train_label_counter = Counter(labels)
    for k, v in train_label_counter.items():
        print('%d: %d' % (k, v))
    return train_label_counter


if __name__ == '__main__':
    setting_name = r'with_bpm_st'
    print('record num:')
    print(db.OsuDB(prepare_data.DEFAULT_TRAIN_MEL_DB_PATH).record_num('FILTERED'))
    dataset = recur_dataset.RecurDataset(
        r'./resources/data/fit/recur/%s' % setting_name,
        db_path=prepare_data.DEFAULT_TRAIN_MEL_DB_PATH,
        audio_dir=prepare_data.DEFAULT_TRAIN_MEL_AUDIO_DIR,
        label_snap=64, audio_mel=16, snap_mel=4, snap_divisor=8,
        take_first=None, max_beatmap_sample_num=50, shuffle=True, keep_label_proportion=True,
        random_seed=DEFAULT_RANDOM_SEED,
    )
    # dataset.logger.setLevel(logging.DEBUG)
    # dataset.prepare()
    # dataset.div_folds(folds=5, save_first=1)
    svc = LinearSVC(random_state=0, tol=1e-5, max_iter=100000)
    fold = 1
    train_data, train_label = dataset.get_fold_items(fold, 'train')
    count_labels(train_label)
    print('train_data.shape')
    print(train_data.shape)
    print('train_label.shape')
    print(train_label.shape)
    svc.fit(train_data, train_label)
    test_data, test_label = dataset.get_fold_items(fold, 'test')
    count_labels(test_label)
    print('test_data.shape')
    print(test_data.shape)
    print('test_label.shape')
    print(test_label.shape)
    test_pred = svc.predict(test_data)
    print(accuracy_score(test_label, test_pred))
    print(confusion_matrix(test_label, test_pred))
    model_save_dir = os.path.join(DEFAULT_MODEL_SAVE_DIR, 'svc')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    with open(os.path.join(model_save_dir, '%s.pkl' % setting_name), 'wb') as f:
        pickle.dump(svc.get_params(), f)
