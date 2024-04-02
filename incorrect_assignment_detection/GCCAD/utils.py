import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import multiprocessing
from sklearn.metrics import roc_auc_score, auc, roc_curve,average_precision_score
from torch_geometric.utils import add_self_loops, degree, softmax, to_dense_adj, dense_to_sparse
from operator import itemgetter
from scipy import sparse
import random

def MAPs(label_lists, score_lists):
    assert len(label_lists) == len(score_lists)
    total_ap = 0
    total_auc = 0
    total_count = 0
    total_outliers = 0
    for sub_labels, sub_scores in zip(label_lists, score_lists):
        assert sub_labels.shape[0] == sub_scores.shape[0]
        auc = roc_auc_score(sub_labels,sub_scores)
        ap = average_precision_score(sub_labels,sub_scores)
        w = (1-sub_labels).sum()
        total_ap += w * ap
        total_auc += w * auc
        total_count += len(sub_labels)
        total_outliers += w 

    mAP = total_ap/total_outliers
    AUC = total_auc/total_outliers

    return AUC, mAP

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
