import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from tqdm import tqdm


def calc_curve_bootstrap(curve, metric, y, preds, n_bootstrap, seed=12345, stratified=True, alpha=95):
    """
    Parameters
    ----------
    curve : function
        Function, which computes the curve.
    metric : fucntion
        Metric to compute, e.g. AUC for ROC curve or AP for PR curve
    y : numpy.array
        Ground truth
    preds : numpy.array
        Predictions
    n_bootstrap:
        Number of bootstrap samples to draw
    seed : int
        Random seed
    stratified : bool
        Whether to do a stratified bootstrapping
    alpha : float
        Confidence intervals width

    """

    np.random.seed(seed)
    metric_vals = []
    ind_pos = np.where(y == 1)[0]
    ind_neg = np.where(y == 0)[0]

    for _ in tqdm(range(n_bootstrap), total=n_bootstrap, desc='Bootstrap:'):
        if stratified:
            ind_pos_bs = np.random.choice(ind_pos, ind_pos.shape[0])
            ind_neg_bs = np.random.choice(ind_neg, ind_neg.shape[0])
            ind = np.hstack((ind_pos_bs, ind_neg_bs))
        else:
            ind = np.random.choice(y.shape[0], y.shape[0])

        if y[ind].sum() == 0:
            continue
        metric_vals.append(metric(y[ind], preds[ind]))

    metric_val = metric(y, preds)
    x_curve_vals, y_curve_vals, _ = curve(y, preds)
    ci_l = np.percentile(metric_vals, (100 - alpha) // 2)
    ci_h = np.percentile(metric_vals, alpha + (100 - alpha) // 2)

    return metric_val, ci_l, ci_h, x_curve_vals, y_curve_vals


def roc_curve_bootstrap(y, preds, savepath=None, n_bootstrap=1000, seed=42, return_curve=False):
    """Evaluates ROC curve using bootstrapping

    Also reports confidence intervals and prints them.

    Parameters
    ----------
    y : numpy.array
        Ground truth
    preds : numpy.array
        Predictions
    savepath: str
        Where to save the figure with ROC curve
    n_bootstrap:
        Number of bootstrap samples to draw
    seed : int
        Random seed

    """
    auc, ci_l, ci_h, fpr, tpr = calc_curve_bootstrap(roc_curve, roc_auc_score, y, preds,
                                                     n_bootstrap, seed, stratified=False, alpha=95)

    plt.figure(figsize=(8, 8))
    plt.title(f'AUC {np.round(auc, 2):.2f} 95% CI [{np.round(ci_l, 2):.2f}-{np.round(ci_h, 2):.2f}]')
    plt.plot(fpr, tpr, 'r-')
    plt.plot([0, 1], [0, 1], '--', color='black')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()
    plt.close()

    print('AUC:', np.round(auc, 5))
    print(f'CI [{ci_l:.5f}, {ci_h:.5f}]')
    if return_curve:
        return auc, ci_l, ci_h, fpr, tpr
    return auc, ci_l, ci_h


def compare_curves(y, preds1, preds2, savepath_roc=None, savepath_pr=None, n_bootstrap=2000, seed=12345):
    plt.figure(figsize=(8, 8))
    auc, ci_l, ci_h, fpr, tpr = calc_curve_bootstrap(roc_curve, roc_auc_score, y, preds1, n_bootstrap=n_bootstrap,
                                                     seed=seed, stratified=True, alpha=95)
    print(f'AUC (method 1): {np.round(auc, 2):.2f} | 95% CI [{np.round(ci_l, 2):.2f},{np.round(ci_h, 2):.2f}]')
    plt.plot(fpr, tpr, 'b-')

    auc, ci_l, ci_h, fpr, tpr = calc_curve_bootstrap(roc_curve, roc_auc_score, y, preds2, n_bootstrap=n_bootstrap,
                                                     seed=seed, stratified=True, alpha=95)
    print(f'AUC (method 2): {np.round(auc, 2):.2f} | 95% CI [{np.round(ci_l, 2):.2f},{np.round(ci_h, 2):.2f}]')
    plt.plot(fpr, tpr, 'r-')
    plt.plot([0, 1], [0, 1], '-', color='black')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.tight_layout()
    if savepath_roc:
        plt.savefig(savepath_roc, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.axhline(y=y.sum() / y.shape[0], color='black', linestyle='--')
    AP, ci_l, ci_h, precision, recall = calc_curve_bootstrap(precision_recall_curve, average_precision_score, y,
                                                             preds1, n_bootstrap, seed, stratified=True, alpha=95)
    print(f'AP (method 1): {np.round(AP, 2):.2f} | 95% CI [{np.round(ci_l, 2):.2f},{np.round(ci_h, 2):.2f}]')

    plt.plot(recall, precision, 'b-')
    AP, ci_l, ci_h, precision, recall = calc_curve_bootstrap(precision_recall_curve, average_precision_score, y,
                                                             preds2, n_bootstrap, seed, stratified=True, alpha=95)
    print(f'AP (method 2): {np.round(AP, 2):.2f} | 95% CI [{np.round(ci_l, 2):.2f},{np.round(ci_h, 2):.2f}]')
    plt.plot(recall, precision, 'r-')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    if savepath_pr:
        plt.savefig(savepath_pr, bbox_inches='tight')
    plt.show()
    plt.close()


# AUC comparison adapted from https://github.com/yandexdataschool/roc_comparison
# Commit e9e3273
#
# Initially adopted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m] * (tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count, _ = compute_ground_truth_statistics(ground_truth, None)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, None)
    return calc_pvalue(aucs, delongcov)
