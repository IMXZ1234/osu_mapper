import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle


def cal_criteria(pred_array, labels_array, target_label_list):
    """labels should be unsigned int"""
    # print(labels_array.size)
    # calculate overall accuracy
    correct_num = 0
    for i in range(labels_array.size):
        if pred_array[i] == labels_array[i]:
            correct_num = correct_num + 1
    accuracy = float(correct_num) / labels_array.size
    # calculate precision, recall and f1 score
    label_num = len(target_label_list)
    TP = np.zeros(label_num)
    FP = np.zeros(label_num)
    FN = np.zeros(label_num)
    TN = np.zeros(label_num)
    for i in range(labels_array.size):
        true_label = labels_array[i]
        pred_label = pred_array[i]
        if true_label == pred_label:
            TP[true_label] = TP[true_label] + 1
            for j in range(label_num):
                if j != true_label:
                    TN[true_label] = TN[true_label] + 1
        else:
            FP[pred_label] = FP[pred_label] + 1
            FN[true_label] = FN[true_label] + 1
            for j in range(label_num):
                if j != true_label and j != pred_label:
                    TN[true_label] = TN[true_label] + 1
    print('each column represents a label, left to right: N,O,A,~')
    print('TP', end=': ')
    print(TP)
    print('FP', end=': ')
    print(FP)
    print('TN', end=': ')
    print(TN)
    print('FN', end=': ')
    print(FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * TP / (2 * TP + FP + FN)
    # if (TP + FP).tolist().count(0) == 0:
    #     precision = TP / (TP + FP)
    # else:
    #     precision = np.zeros(label_num)
    #     print('precision doesn\'t exist')
    # if (TP + FN).tolist().count(0) == 0:
    #     recall = TP / (TP + FN)
    # else:
    #     recall = 0
    #     print('recall doesn\'t exist')
    # if (2 * TP + FP + FN).tolist().count(0) == 0:
    #     f1 = 2 * TP / (2 * TP + FP + FN)
    # else:
    #     f1 = 0
    #     print('f1 doesn\'t exist')
    return precision, recall, f1, accuracy


def get_class_TPR_FPR_vector_array(pred_prob_array_in, labels_array):
    # print(labels_array)
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
    TP_matrix_array = np.zeros([label_num, sample_num + 1])
    FP_matrix_array = np.zeros([label_num, sample_num + 1])
    # micro way of calculation
    for i in range(label_num):
        # a line of ROC_matrix represents ROC y-value for a specific label
        TP = 0
        FP = 0
        # label_pred_prob = pred_prob_array[:, i]
        for k in range(1, sample_num + 1):
            prob_threshold = sorted_array_labels[sample_num - k, i]
            for j in range(sample_num):
                # prediction is true according to current threshold
                if pred_prob_array[j, i] >= prob_threshold:
                    # set to -inf
                    pred_prob_array[j, i] = sorted_array_labels[0, i] - 1
                    if labels_matrix_array[j, i] == 1:
                        TP = TP + 1
                    else:
                        FP = FP + 1
                    break
            TP_matrix_array[i, k] = TP
            FP_matrix_array[i, k] = FP
        TP_matrix_array[i, 0] = 0
        FP_matrix_array[i, 0] = 0
    return TP_matrix_array, FP_matrix_array


def get_TPR_FPR_vector_array(pred_prob_array, labels_array, cal_method):
    print(pred_prob_array)
    print(labels_array)
    label_num = pred_prob_array.shape[1]
    sample_num = pred_prob_array.shape[0]
    print('label_num', label_num)
    print('sample_num', sample_num)
    labels_matrix_array = np.zeros(pred_prob_array.shape)
    # get label matrix according to labels_array
    for i in range(sample_num):
        labels_matrix_array[i, labels_array[i]] = 1
    # print(labels_matrix_array)
    # sort for each label
    sorted_array_labels = np.sort(pred_prob_array, 0)
    # sort for all labels
    sorted_array_all = np.sort(pred_prob_array.reshape(-1, 1), 0)
    TP_matrix_array = np.zeros([label_num, sample_num + 1])
    TP_vector_array = np.zeros([1, sample_num * label_num + 1])
    FP_matrix_array = np.zeros([label_num, sample_num + 1])
    FP_vector_array = np.zeros([1, sample_num * label_num + 1])
    if cal_method == 'macro':
        # macro way of calculation
        for k in range(sample_num * label_num):
            TP = 0
            FP = 0
            prob_threshold = sorted_array_all[sample_num * label_num - k - 1]
            for i in range(label_num):
                for j in range(sample_num):
                    # prediction is true according to current threshold
                    if pred_prob_array[j, i] > prob_threshold:
                        if labels_matrix_array[j, i] == 1:
                            TP = TP + 1
                        else:
                            FP = FP + 1
            TP_vector_array[0, k] = TP
            FP_vector_array[0, k] = FP
        TP_vector_array[0, sample_num * label_num] = sample_num
        FP_vector_array[0, sample_num * label_num] = sample_num * (label_num - 1)
        # deal with condition that some predicted probabilities are the same
        # which will cause (TP_vector_array[i], FP_vector_array[i]) remain unchanged
        for i in range(sample_num * label_num):
            # find out length of unchanged sequence
            j = i + 1
            while j <= sample_num * label_num:
                if TP_vector_array[0, i] == TP_vector_array[0, j] and FP_vector_array[0, i] == FP_vector_array[0, j]:
                    j = j + 1
                else:
                    break
            # length of unchanged sequence != 0
            if j != i + 1:
                TP_increase = TP_vector_array[0, j] - TP_vector_array[0, i]
                FP_increase = FP_vector_array[0, j] - FP_vector_array[0, i]
                total_increase = TP_increase + FP_increase
                # TPR increase is considered prior
                k = i + 1
                n = 1
                while n <= TP_increase and total_increase > 1:
                    TP_vector_array[0, k: j] = TP_vector_array[0, k: j] + 1
                    total_increase = total_increase - 1
                    n = n + 1
                    k = k + 1
                m = 1
                while m <= FP_increase and total_increase > 1:
                    FP_vector_array[0, k: j] = FP_vector_array[0, k: j] + m
                    total_increase = total_increase - 1
                    m = m + 1
            # print(TP_vector_array)
            # print(FP_vector_array)
        # print(TP_vector_array)
        # print(FP_vector_array)
        return TP_vector_array, FP_vector_array
    else:
        # micro way of calculation
        for i in range(label_num):
            # a line of ROC_matrix represents ROC y-value for a specific label
            for k in range(sample_num):
                TP = 0
                FP = 0
                prob_threshold = sorted_array_labels[sample_num - k - 1, i]
                for j in range(sample_num):
                    # prediction is true according to current threshold
                    if pred_prob_array[j, i] > prob_threshold:
                        if labels_matrix_array[j, i] == 1:
                            TP = TP + 1
                        else:
                            FP = FP + 1
                TP_matrix_array[i, k] = TP
                FP_matrix_array[i, k] = FP
            TP_matrix_array[i, sample_num] = 1
            FP_matrix_array[i, sample_num] = sample_num - 1
        # print(TP_matrix_array)
        # print(FP_matrix_array)
        # deal with condition that some predicted probabilities are the same
        # which will cause (TP_vector_array[i], FP_vector_array[i]) remain unchanged
        for t in range(label_num):
            for i in range(sample_num):
                # find out length of unchanged sequence
                j = i + 1
                while j <= sample_num:
                    if TP_matrix_array[t, i] == TP_matrix_array[t, j] and FP_matrix_array[t, i] == FP_matrix_array[
                        t, j]:
                        j = j + 1
                    else:
                        break
                # length of unchanged sequence != 0
                if j != i + 1:
                    TP_increase = TP_matrix_array[t, j] - TP_matrix_array[t, i]
                    FP_increase = FP_matrix_array[t, j] - FP_matrix_array[t, i]
                    total_increase = TP_increase + FP_increase
                    # TPR increase is considered prior
                    k = i + 1
                    n = 1
                    while n <= TP_increase and total_increase > 1:
                        TP_matrix_array[t, k: j] = TP_matrix_array[t, k: j] + 1
                        total_increase = total_increase - 1
                        n = n + 1
                        k = k + 1
                    m = 1
                    while m <= FP_increase and total_increase > 1:
                        FP_matrix_array[t, k: j] = FP_matrix_array[t, k: j] + m
                        total_increase = total_increase - 1
                        m = m + 1
        return TP_matrix_array, FP_matrix_array


def plot_class_ROC(TP_vector_array, FP_vector_array, title=None, save_fig=False, save_dir=None, save_fig_name=None):
    Y = TP_vector_array / TP_vector_array[-1]
    X = FP_vector_array / FP_vector_array[-1]
    if title is not None:
        plt.title(title)
    # 设置坐标轴范围
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    x_ticks = np.arange(0, 1, 0.1)
    y_ticks = np.arange(0, 1, 0.1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.plot(X, Y)
    if save_fig:
        if save_dir is None:
            save_dir = os.getcwd()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if save_fig_name is None:
            save_fig_name = title if title is not None else 'ROC Curve'
        plt.savefig(os.path.join(save_dir, save_fig_name))
    plt.show()


def plot_ROC(TP_matrix_array, FP_matrix_array):
    Y = []
    for t in range(TP_matrix_array.shape[0]):
        y = []
        for i in range(TP_matrix_array.shape[1] - 1):
            if TP_matrix_array[t, i] == TP_matrix_array[t, i + 1]:
                y.append(TP_matrix_array[t, i + 1])
        Y.append(y)
    Y = np.array(Y)
    print(Y)
    Y_average = Y.sum(0)
    Y_average = Y_average / TP_matrix_array[0, -1]
    print(Y_average)
    X = np.array(range(Y_average.size))
    X = X / X.size
    print(X)
    plt.scatter(X, Y_average)
    plt.show()
    return True


def cal_AUC(TP_matrix_array, FP_matrix_array):
    area_sum = np.zeros([TP_matrix_array.shape[0], 1])
    for t in range(TP_matrix_array.shape[0]):
        i = 0
        j = 1
        while i < TP_matrix_array.shape[1]:
            while j < TP_matrix_array.shape[1] and TP_matrix_array[t, j] == TP_matrix_array[t, i]:
                j = j + 1
            area_sum[t, 0] = area_sum[t, 0] + (j - i - 1) * TP_matrix_array[t, i]
            i = j
            j = i + 1
    area_sum = area_sum / TP_matrix_array[:, -1] / FP_matrix_array[:, -1]
    return sum(area_sum) / TP_matrix_array.shape[0]


def cal_class_AUC(TP_vector_array, FP_vector_array):
    current_Y = 0
    sum = 0
    for i in range(1, len(TP_vector_array)):
        if TP_vector_array[i] != TP_vector_array[i - 1]:
            current_Y += 1
        else:
            sum += current_Y
    return sum / TP_vector_array[-1] / FP_vector_array[-1]
