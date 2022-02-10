import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle

from data import data_const, data_get


def get_class_precision_recall_array(pred_prob_array_in, labels_array):
    pred_prob_array = pred_prob_array_in.copy()
    # print(pred_prob_array)
    # print(labels_array)
    label_num = pred_prob_array.shape[1]
    sample_num = pred_prob_array.shape[0]
    # print('label_num', label_num)
    # print('sample_num', sample_num)
    labels_matrix_array = np.zeros(pred_prob_array.shape)
    # get label matrix according to labels_array
    for i in range(sample_num):
        labels_matrix_array[i, labels_array[i]] = 1
    # print(labels_matrix_array)
    # sort for each label
    sorted_array_labels = np.sort(pred_prob_array, 0)
    # sort for all labels
    precision_array = np.zeros([label_num, sample_num])
    recall_array = np.zeros([label_num, sample_num])
    class_true_pos_sample_num = np.zeros(label_num)
    class_pred_pos_sample_num = np.zeros(label_num)
    pred_labels_array = np.argmax(pred_prob_array, axis=1)
    for i in range(sample_num):
        class_true_pos_sample_num[labels_array[i]] += 1
        class_pred_pos_sample_num[pred_labels_array[i]] += 1
    for i in range(label_num):
        # a line of ROC_matrix represents ROC y-value for a specific label
        TP = 0
        P = 0
        # label_pred_prob = pred_prob_array[:, i]
        for k in range(0, sample_num):
            prob_threshold = sorted_array_labels[sample_num - k - 1, i]
            for j in range(sample_num):
                # prediction is true according to current threshold
                if pred_prob_array[j, i] >= prob_threshold:
                    # set to -inf
                    pred_prob_array[j, i] = sorted_array_labels[0, i] - 1
                    if labels_matrix_array[j, i] == 1:
                        TP += 1
                    P += 1
                    break
            precision_array[i, k] = TP / P
            recall_array[i, k] = TP
        recall_array[i, :] /= class_true_pos_sample_num[i]
    return precision_array, recall_array


def plot_class_PR(precision_vector_array, recall_vector_array, title=None, save_fig=False, save_dir=None, save_fig_name=None, plot_fig=True):
    # print('X')
    # print(X)
    # print('Y')
    # print(Y)
    # plt.scatter(X, Y)
    if title is not None:
        plt.title(title)
    # 设置坐标轴范围
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    x_ticks = np.arange(0, 1, 0.1)
    y_ticks = np.arange(0, 1, 0.1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.plot(recall_vector_array, precision_vector_array)
    if save_fig:
        if save_dir is None:
            save_dir = os.getcwd()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if save_fig_name is None:
            save_fig_name = title if title is not None else 'PR Curve'
        plt.savefig(os.path.join(save_dir, save_fig_name))
    if plot_fig:
        plt.show()
    else:
        plt.clf()


def cal_class_AUPRC(precision_vector_array, recall_vector_array):
    auprc = precision_vector_array[0]
    num = 0
    for i in range(1, len(precision_vector_array)):
        if recall_vector_array[i] != recall_vector_array[i - 1]:
            auprc += precision_vector_array[i]
            num += 1
    return auprc / num


def get_epoch_AUPRC(result_dir, data_dir, fold, epoch, is_test=True):
    score_file_name_base = 'epoch%d_test_score.pkl' if is_test else 'epoch%d_train_score.pkl'
    with open(os.path.join(result_dir,
                           'training-result%d' % fold,
                           score_file_name_base % epoch),
              'rb') as score_file:
        score_dict = pickle.load(score_file)
    train_score_dict_sample_name = list(score_dict.keys())
    # print('train_score_dict_sample_name')
    # print(train_score_dict_sample_name)
    train_score = [score_dict[train_score_dict_sample_name[i]] for i in range(len(train_score_dict_sample_name))]
    # print(train_score)
    # print('train_score')
    train_label, _ = data_get.get_label_list_by_name_list(
        train_score_dict_sample_name, data_dir, fold, is_test=is_test)
    # print(train_label)
    precision_array, recall_array = get_class_precision_recall_array(
        np.stack(train_score, axis=0), train_label)
    # auc = cal_AUC(TP_matrix_array, FP_matrix_array)
    auprc = [cal_class_AUPRC(precision_array[i], recall_array[i]) for i in range(precision_array.shape[0])]
    return auprc, precision_array, recall_array


def get_epoch_AUPRC_all_folds(result_dir, data_dir, folds_num, epoch, is_test=True):
    auprc, precision_array_list, recall_matrix_array_list = [], [], []
    for fold_index in range(folds_num):
        fold_auc, fold_precision_array, fold_recall_array = get_epoch_AUPRC(result_dir, data_dir, fold_index + 1, epoch, is_test)
        auprc.append(fold_auc)
        precision_array_list.append(fold_precision_array)
        recall_matrix_array_list.append(fold_recall_array)
    class_num = len(auprc[0])
    fold_avg_auprc = [0 for _ in range(class_num)]
    for class_index in range(class_num):
        for fold_index in range(folds_num):
            fold_avg_auprc[class_index] += auprc[fold_index][class_index]
        fold_avg_auprc[class_index] /= folds_num
    return auprc, precision_array_list, recall_matrix_array_list, fold_avg_auprc

