import os
import numpy as np
from termcolor import colored

from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix, mean_squared_error
from oaprogression.kvs import GlobalKVS


def log_metrics(boardlogger, train_loss, val_loss, gt_progression, preds_progression, gt_kl, preds_kl):
    kvs = GlobalKVS()

    # Computing Validation metrics
    preds_progression_bin = preds_progression[:, 1:].sum(1)
    preds_kl_bin = preds_kl[:, 1:].sum(1)

    cm_prog = confusion_matrix(gt_progression, preds_progression.argmax(1))
    cm_kl = confusion_matrix(gt_kl, preds_kl.argmax(1))

    kappa_prog = cohen_kappa_score(gt_progression, preds_progression.argmax(1), weights="quadratic")
    acc_prog = np.mean(cm_prog.diagonal().astype(float) / cm_prog.sum(axis=1))
    mse_prog = mean_squared_error(gt_progression, preds_progression.argmax(1))
    auc_prog = roc_auc_score(gt_progression > 0, preds_progression_bin)

    kappa_kl = cohen_kappa_score(gt_kl, preds_kl.argmax(1), weights="quadratic")
    acc_kl = np.mean(cm_kl.diagonal().astype(float) / cm_kl.sum(axis=1))
    mse_kl = mean_squared_error(gt_kl, preds_kl.argmax(1))
    auc_oa = roc_auc_score(gt_kl > 1, preds_kl_bin)

    res = {
     'epoch': kvs['cur_epoch'],
     'val_loss': val_loss,
     'auc_prog': auc_prog,
     'kappa_prog': kappa_prog,
     'acc_prog': acc_prog,
     'mse_prog': mse_prog,
     'auc_oa': auc_oa,
     'kappa_kl': kappa_kl,
     'acc_kl': acc_kl,
     'mse_kl': mse_kl,
     'cm_prog': cm_prog,
     'cm_kl': cm_kl,
     }

    print(colored('====> ', 'green') + f'Train loss: {train_loss:.5f}')
    print(colored('====> ', 'green') + f'Validation loss: {val_loss:.5f}')
    print(colored('====> ', 'green') + f'Validation AUC [prog]: {auc_prog:.5f}')
    print(colored('====> ', 'green') + f'Validation AUC [oa]: {auc_oa:.5f}')
    print(colored('====> ', 'green') + f'Kappa [oa]: {kappa_kl:.5f}')

    boardlogger.add_scalars('Losses', {'train': train_loss, 'val': val_loss}, kvs['cur_epoch'])
    boardlogger.add_scalars('AUC progression', {'val': auc_prog}, kvs['cur_epoch'])

    kvs.update(f'losses_fold_[{kvs["cur_fold"]}]', {'epoch': kvs['cur_epoch'],
                                                    'train_loss': train_loss,
                                                    'val_loss': val_loss})

    kvs.update(f'val_metrics_fold_[{kvs["cur_fold"]}]', res)

    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))
