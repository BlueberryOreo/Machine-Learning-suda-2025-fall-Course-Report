# Evaluation utilities for NMI, ARI and ACC

import sklearn.metrics as metrics
from scipy.optimize import linear_sum_assignment
import numpy as np
from numpy.typing import ArrayLike

def nmi(true_labels: ArrayLike, pred_labels: ArrayLike) -> float:
    """Compute Normalized Mutual Information (NMI)"""
    return metrics.normalized_mutual_info_score(true_labels, pred_labels)

def ari(true_labels: ArrayLike, pred_labels: ArrayLike) -> float:
    """Compute Adjusted Rand Index (ARI)"""
    return metrics.adjusted_rand_score(true_labels, pred_labels)

def cluster_acc(true_labels: ArrayLike, pred_labels: ArrayLike) -> float:
    """
    Compute clustering accuracy (cluster-acc) with optimal label mapping.

    Args:
        true_labels: ArrayLike, shape (n_samples,)
        pred_labels: ArrayLike, shape (n_samples,)

    Returns:
        acc: float
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    if true_labels.shape[0] != pred_labels.shape[0]:
        raise ValueError("true_labels and pred_labels must have the same length.")

    # 映射到连续的 0...K-1，避免标签不是从0开始/不连续的情况
    true_ids, true_mapped = np.unique(true_labels, return_inverse=True)
    pred_ids, pred_mapped = np.unique(pred_labels, return_inverse=True)

    n_true = len(true_ids)
    n_pred = len(pred_ids)
    n = max(n_true, n_pred)

    # 构建混淆矩阵（计数矩阵）
    w = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(true_mapped, pred_mapped):
        w[t, p] += 1

    # 匈牙利算法求最大匹配（linear_sum_assignment 是最小化，所以用 max-w）
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    correct = w[row_ind, col_ind].sum()
    acc = correct / true_labels.shape[0]
    return float(acc)

def evaluate_clustering(
    true_labels: ArrayLike,
    pred_labels: ArrayLike,
) -> dict:
    """
    Evaluate clustering performance using NMI, ARI and ACC.

    Args:
        true_labels: ArrayLike, shape (n_samples,)
        pred_labels: ArrayLike, shape (n_samples,)
    Returns:
        results: dict with keys "NMI", "ARI", "ACC"
    """
    results = {
        "NMI": nmi(true_labels, pred_labels),
        "ARI": ari(true_labels, pred_labels),
        "ACC": cluster_acc(true_labels, pred_labels),
    }
    return results