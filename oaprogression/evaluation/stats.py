import numpy as np
from scipy import interp
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm


def roc_curve_bootstrap(y, preds, savepath=None, n_bootstrap=1000, seed=42):
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
    auc = roc_auc_score(y, preds)
    np.random.seed(seed)
    aucs = []
    tprs = []
    base_fpr = np.linspace(0, 1, 1001)
    for _ in tqdm(range(n_bootstrap), total=n_bootstrap, desc='Bootstrap:'):
        ind = np.random.choice(y.shape[0], y.shape[0])
        if y[ind].sum() == 0:
            continue
        aucs.append(roc_auc_score(y[ind], preds[ind]))
        fpr, tpr, _ = roc_curve(y[ind], preds[ind])
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = np.mean(tprs, 0)
    std = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    CI_l, CI_h = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

    plt.figure(figsize=(6, 6))
    plt.title(f'AUC {auc:.5f} 95% CI [{CI_l:.5f}-{CI_h:.5f}]')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
    plt.plot(base_fpr, mean_tprs, 'r-')
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
    print(f'CI [{CI_l:.5f}, {CI_h:.5f}]')
    return auc, CI_l, CI_h


