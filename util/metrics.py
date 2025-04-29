""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""


import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from collections import namedtuple

from util.custom_datasets import new_supk_dict, new_phyl_dict, new_genus_dict

from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve

from sklearn.metrics import confusion_matrix, precision_score

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve  #, auc
from sklearn.preprocessing import label_binarize
from scipy.stats import mode

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    # output = F.softmax(output.clone().detach().float(), dim=1)  # F.softmax(torch.tensor(output).float(), dim=1)
    
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.shape[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def get_result_rank(results, topk_list):
    topk_res = []
    for topk in topk_list:
        a = mode(np.array(results[:,:topk]).transpose())[0]
        topk_res.append(a)
    return topk_res


def macro_average_precision(y_pred, y_true):
    # 获取所有类别
    classes = set(list(y_true.cpu().numpy())) 
    # 初始化宏平均精度
    macro_avg_precision = 0.0
    macro_avg_recall = 0.0
    macro_avg_f1 = 0.0
    # 计算每个类别的精度并累加
    for cls in classes:
        y_true_cls = [1 if label == cls else 0 for label in y_true.cpu()]
        y_pred_cls = [1 if label == cls else 0 for label in y_pred]

        # 计算混淆矩阵
        out = confusion_matrix(y_true_cls, y_pred_cls).ravel()
        if out.shape[0] == 1:
            fp = tn = fn = 0
            tp = len(y_true)
        else:
            tn, fp, fn, tp = out

        # 计算精度
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # 累加精度
        macro_avg_precision += precision
        macro_avg_recall += recall
        macro_avg_f1 += f1

    # 计算宏平均精度
    macro_avg_precision /= len(classes)  #ttp / (ttp + tfp)
    macro_avg_recall /= len(classes)
    macro_avg_f1 /= len(classes)

    
    return macro_avg_precision * 100.0, macro_avg_recall * 100, macro_avg_f1 * 100


# def new_macro_average_precision(y_pred, y_true):
#     # 获取所有类别
#     classes = set(list(y_true.cpu().numpy()))  #.union(set(y_pred.numpy()))
#     y_true_ohe = torch.nn.functional.one_hot(y_true, num_classes=int(y_pred.shape[1]))
#     macro_avg2 = precision_score(y_true, np.argmax(y_pred, axis=-1), average='macro')
#     print("precision_score 2  ", macro_avg2)
#     macro_avg_precision = average_precision_score(y_true_ohe, y_pred, average="macro")
#     return macro_avg_precision * 100.0  #, macro_avg_recall * 100, macro_avg_f1 * 100


def weighted_macro_average_precision(y_pred, y_true):
    # 获取所有类别
    classes = set(list(y_true.cpu().numpy()))  
    # 初始化宏平均精度
    weighted_macro_avg_precision = 0.0
    # 计算每个类别的精度并累加
    for cls in classes:
        y_true_cls = [1 if label == cls else 0 for label in y_true.cpu()]
        y_pred_cls = [1 if label == cls else 0 for label in y_pred]

        # 计算混淆矩阵
        if confusion_matrix(y_true_cls, y_pred_cls).ravel().shape[0] == 1:
            fp = tn = fn = 0
            tp = len(y_true)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true_cls, y_pred_cls).ravel()

        # 计算精度
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # 累加精度
        weighted_macro_avg_precision += (precision * sum(y_true_cls) / len(y_true))

    return weighted_macro_avg_precision * 100.0


def micro_average_precision(y_pred, y_true):
    # 获取所有类别
    
    classes = set(list(y_true.cpu().numpy()))
    # 初始化宏平均精度
    micro_avg_precision = 0.0
    ttp = 0.0
    ttn = 0.0
    tfp = 0.0
    tnp = 0.0
    # 计算每个类别的精度并累加
    for cls in classes:
        y_true_cls = [1 if label == cls else 0 for label in y_true.cpu()]
        y_pred_cls = [1 if label == cls else 0 for label in y_pred]

        # 计算混淆矩阵
        if confusion_matrix(y_true_cls, y_pred_cls).ravel().shape[0] == 1:
            fp = tn = fn = 0
            tp = len(y_true)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true_cls, y_pred_cls).ravel()
        # 计算精度
        ttp += tp
        tfp += fp

    # 计算宏平均精度
    micro_avg_precision = ttp / (ttp + tfp)
    return micro_avg_precision * 100.0


def weighted_micro_average_precision(y_pred, y_true):
    # 获取所有类别
    classes = set(list(y_true.cpu().numpy()))  
    
    # 初始化宏平均精度
    weighted_micro_avg_precision = 0.0
    ttp = 0.0
    ttn = 0.0
    tfp = 0.0
    tnp = 0.0
    # 计算每个类别的精度并累加
    for cls in classes:
        y_true_cls = [1 if label == cls else 0 for label in y_true.cpu()]
        y_pred_cls = [1 if label == cls else 0 for label in y_pred]

        # 计算混淆矩阵
        if confusion_matrix(y_true_cls, y_pred_cls).ravel().shape[0] == 1:
            fp = tn = fn = 0
            tp = len(y_true)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true_cls, y_pred_cls).ravel()
        # 计算精度
        ttp += (tp * sum(y_true_cls) / len(y_true))
        tfp += (fp * sum(y_true_cls) / len(y_true))

    # 计算宏平均精度
    weighted_micro_avg_precision = ttp / (ttp + tfp)
    return weighted_micro_avg_precision * 100.0




def compute_roc(args, preds, trues):
    """Computes FPR, TPR, ROC AUC.

    Parameters:
    - trues: Either ndarray of shape (n, #classes), list of class labels, or list of class indices.
    - preds: Numpy array of shape (n, #classes).
    - classes: List of classes.

    Returns:
    - roc_info: Named tuple containing FPR, TPR, and ROC AUC.
    """
    # preds = np.array(preds)
    # trues = np.array(trues)
    classes = preds.shape[1]  # set(list(trues.cpu().numpy()))  #
    # trues = trues
    # preds = F.softmax(preds.float(), dim=1).numpy()  # 应用 softmax 转换

    # y = trues
    fpr = {}
    tpr = {}
    roc_auc = {}
    y = label_binarize(trues, classes=range(classes))
    for i, c in enumerate(range(classes)):
        if np.any(y[:, i]):
            fpr[c], tpr[c], _ = roc_curve(y[:, i], preds[:, i])
            roc_auc[c] = auc(fpr[c], tpr[c])

    # 计算微平均的 ROC 曲线和 ROC 面积
    fpr['micro'], tpr['micro'], _ = roc_curve(y.ravel(), preds.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # 聚合所有假正例率
    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(classes) if np.any(y[:, c])]))

    # 在这些点上插值所有 ROC 曲线
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(classes):
        if np.any(y[:, c]):  # 只在类别存在时计算 ROC 曲线
            mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])

    # 最后取平均并计算 AUC
    mean_tpr /= len(set(list(trues.cpu().numpy())))

    # fpr['macro'] = all_fpr
    # tpr['macro'] = mean_tpr
    # roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    return namedtuple('ROC', 'fpr, tpr, roc_auc')(fpr, tpr, roc_auc)


def plot_roc(args, rank, trues, preds, all_curves=True):
    preds = F.softmax(preds.float(), dim=1)
    
    fpr, tpr, roc_auc = compute_roc(args, preds, trues)
    # classes = set(list(trues.cpu().numpy()))  # preds.size(1)
    all_class = set(list(trues.cpu().numpy()))
    classes = set([t for t in set(list(trues.cpu().numpy())) if list(trues.cpu().numpy()).count(t) > 1500])

    plt.figure()
    plt.plot(fpr['micro'], tpr['micro'], label=f'Average (area = {roc_auc["micro"]:.2f})', linestyle=(':' if all_curves else '-'), linewidth=4)

    # plt.plot(fpr['macro'], tpr['macro'],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc['macro']),
    #          color='navy', linestyle=':', linewidth=4)
    # if len(classes) > 10:
    #     all_curves = False
    if (all_curves):
        if rank == "supk":
            for c in classes:
                plt.plot(fpr[c], tpr[c], label=f'{new_supk_dict[c]} (area = {roc_auc[c]:.2f})')
        elif rank == "genus":
            for c in list(all_class)[:7]:
                plt.plot(fpr[c], tpr[c], label=f'{new_genus_dict[c]} (area = {roc_auc[c]:.2f})')
        elif rank == "phyl":
            for c in list(classes)[:7]:
                plt.plot(fpr[c], tpr[c], label=f'{new_phyl_dict[c]} (area = {roc_auc[c]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC on {} '.format(rank, args.data))
    if (all_curves):
        plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(args.output_dir + "/roc_{}.jpg".format(rank))
    plt.close()
    return roc_auc



def compute_pr(args, trues, preds):
    """Computes Precision, Recall, and PR AUC.

    Parameters:
    - trues: Either ndarray of shape (n, #classes), list of class labels, or list of class indices.
    - preds: Numpy array of shape (n, #classes).

    Returns:
    - pr_info: Named tuple containing Precision, Recall, and PR AUC.
    """
    classes = preds.shape[1]
    # preds = F.softmax(torch.tensor(preds).float(), dim=1).numpy()  # 使用PyTorch中的softmax进行转换

    y = label_binarize(trues, classes=range(classes))
    precision = {}
    recall = {}
    pr_auc = {}

    for i in range(classes):
        if np.any(y[:, i]):
            precision[i], recall[i], _ = precision_recall_curve(y[:, i], preds[:, i])
            pr_auc[i] = auc(recall[i], precision[i])

    # 计算微平均的 PR 曲线和 PR 面积
    precision['micro'], recall['micro'], _ = precision_recall_curve(y.ravel(), preds.ravel())
    pr_auc['micro'] = auc(recall['micro'], precision['micro'])

    # 聚合所有召回率
    all_recall = np.unique(np.concatenate([recall[i] for i in range(classes) if np.any(y[:, i])]))

    # 在这些点上插值所有 PR 曲线
    mean_precision = np.zeros_like(all_recall)
    for i in range(classes):
        if np.any(y[:, i]):  # 只在类别存在时计算 PR 曲线
            mean_precision += np.interp(all_recall, recall[i], precision[i])

    # 最后取平均并计算 PR AUC
    mean_precision /= len(set(list(trues.cpu().numpy())))

    return namedtuple('PR', 'precision, recall, pr_auc, mean_precision')(precision, recall, pr_auc, mean_precision)


# 绘制 PR 曲线
def plot_pr_curve(args, rank, trues, preds):
    preds = F.softmax(preds.float(), dim=1)
    pr_info = compute_pr(args, trues, preds)
    # classes = range(len(pr_info.precision) - 1)  # 不包括 'micro'
    # if rank == "genus":
    all_class = set(list(trues.cpu().numpy()))
    classes = set([t for t in all_class if list(trues.cpu().numpy()).count(t) > 1500])
    # 绘制每个类别的 PR 曲线
    all_curves = True
    # if len(classes) > 10:
    #     all_curves = False
    # if (all_curves):
    if rank == "supk":
        for i in classes:
            plt.plot(pr_info.recall[i], pr_info.precision[i], label=f'{new_supk_dict[i]}')
    elif rank == "genus":
        for i in list(all_class)[:7]:
            plt.plot(pr_info.recall[i], pr_info.precision[i], label=f'{new_genus_dict[i]}')
    elif rank == "phyl":
        for i in list(classes)[:7]:
            plt.plot(pr_info.recall[i], pr_info.precision[i], label=f'{new_phyl_dict[i]}')

    # 绘制微平均的 PR 曲线
    plt.plot(pr_info.recall['micro'], pr_info.precision['micro'], label='Micro-average')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('{} Precision-Recall Curve on {}'.format(rank, args.data))
    plt.savefig(args.output_dir + "/PR_{}.jpg".format(rank))
    plt.close()


# Example usage:
# Replace 'output' and 'target' with your model's output and ground truth labels
# output should be a NumPy array, and target should be a list or NumPy array of class indices
# roc_result = compute_roc(target, output, classes)
