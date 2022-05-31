import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.GnBu, title=None, save_fig=False, save_fig_dir=None, plot_fig=True):  # PuBu
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')
    # print(title)
    class_sample_num = np.sum(cm, axis=1)
    class_acc = []
    for i in range(cm.shape[0]):
        class_acc.append(cm[i, i] / class_sample_num[i])
    # print(class_acc)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    cb = plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 family="Times New Roman", fontsize=18)

    cb.ax.tick_params(labelsize=17)  # 设置colorbar刻度字体大小
    labels = cb.ax.get_xticklabels() + cb.ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    cb.ax.xaxis.set_ticks_position('top')  # 将x轴的位置设置在顶部
    plt.ylabel('True Label', family="Times New Roman", weight="bold", size=20)
    plt.xlabel('Predicted Label', family="Times New Roman", weight="bold", size=20)
    plt.tick_params(labelsize=18)
    labels = cb.ax.get_xticklabels() + cb.ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if save_fig:
        fig_file_name = title + '.jpg' if title is not None else 'confusion_matrix.jpg'
        if not os.path.exists(save_fig_dir):
            os.makedirs(save_fig_dir)
        plt.savefig(os.path.join(save_fig_dir, fig_file_name))
    if plot_fig:
        plt.show()
    else:
        plt.clf()


if __name__ == '__main__':
    method = "200frame"

    # true labels
    gt = ['val1_label.pkl', 'val2_label.pkl', 'val3_label.pkl', 'val4_label.pkl', 'val5_label.pkl']
    y_true = []
    y_pred = []
    y_name = []
    y_score_nosoftmax = []
    right_num = total_num = 0
    for agt in gt:
        label = open(os.path.join('./data/original', agt), 'rb')
        label = np.array(pickle.load(label)) # name、label列表构成的元组，转换为ndarray
        fold = agt.split("_")[0].split("l")[-1]
        result_path = "./" + method + "/training-result" + fold
        # label scores
        r = open(os.path.join(result_path, 'epoch70_test_score.pkl'), 'rb')
        r = list(pickle.load(r).item_names()) # name、score元组构成的列表
        for j in range(len(label[0])):
            name1, ls = label[:, j]  # 样本的名字和真实类别
            name2, rs = r[j]  # 预测分数
            assert name1 == name2 # 顺序一致
            y_score_nosoftmax.append(rs.tolist())  # 没有经过softmax的预测概率
            rr = np.argmax(rs)  # 预测类别
            y_name.append(name1)  # 样本名字
            y_pred.append(int(rr))  # 预测标签集合
            y_true.append(int(ls))  # 真实标签集合
            right_num += int(rr == int(ls))  # 预测标签=真实标签
            total_num += 1  # 样本总数
            if int(rr) == 1 and int(ls) == 0:
                print(name1)

    print("预测标签=真实标签 的数量：{}".format(right_num))
    print("总样本 的数量：{}".format(total_num))
    print("总accuracy为：{}".format(right_num / total_num))

    # 混淆矩阵
    C = confusion_matrix(y_true, y_pred, labels=[0, 1])
    f, ax = plt.subplots()
    plot_confusion_matrix(C, classes=['0', '1'], normalize=False)

    n_classes = 2
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = []
    for j in range(len(y_score_nosoftmax)):  # softmax，归一化预测概率值
        z = np.array(y_score_nosoftmax[j])
        y_score.append(np.exp(z) / sum(np.exp(z)))
    y_score = np.array(y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score[:,1])
    roc_auc = auc(fpr, tpr)

    # 画ROC曲线
    lw = 2
    figure, ax = plt.subplots()
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 24}
    plt.plot(fpr, tpr, color='steelblue', lw=lw, marker="x", markersize=4, label='AUC = %0.2f' % roc_auc, linestyle="-")
    plt.legend(loc="lower right", fancybox=True, prop=font1)
    plt.plot([0, 1], [0, 1], color='dimgray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve', family="Times New Roman", weight='black', size=26)
    plt.xlabel('False Positive Rate', family="Times New Roman", weight='bold', size=26)
    plt.ylabel('True Positive Rate', family="Times New Roman", weight='bold', size=26)
    plt.grid(b=True, ls='-.')
    plt.tick_params(labelsize=24)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.tight_layout()
    plt.show()
