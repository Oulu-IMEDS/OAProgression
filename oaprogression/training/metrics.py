import os
from termcolor import colored
from oaprogression.kvs import GlobalKVS
from oaprogression.evaluation import tools


def log_metrics(boardlogger, train_loss, val_loss, gt_progression, preds_progression, gt_kl, preds_kl):
    kvs = GlobalKVS()

    res = tools.calc_metrics(gt_progression, gt_kl, preds_progression, preds_kl)
    res['val_loss'] = val_loss,
    res['epoch'] = kvs['cur_epoch']

    print(colored('====> ', 'green') + f'Train loss: {train_loss:.5f}')
    print(colored('====> ', 'green') + f'Validation loss: {val_loss:.5f}')
    print(colored('====> ', 'green') + f'Validation AUC [prog]: {res["auc_prog"]:.5f}')
    print(colored('====> ', 'green') + f'Validation F1 [prog]: {res["f1_score_prog"]:.5f}')
    print(colored('====> ', 'green') + f'Validation AP [prog]: {res["ap_prog"]:.5f}')

    print(colored('====> ', 'green') + f'Validation AUC [oa]: {res["auc_oa"]:.5f}')
    print(colored('====> ', 'green') + f'Kappa [oa]: {res["kappa_kl"]:.5f}')

    boardlogger.add_scalars('Losses', {'train': train_loss, 'val': val_loss}, kvs['cur_epoch'])
    boardlogger.add_scalars('AUC progression', {'val': res['auc_prog']}, kvs['cur_epoch'])
    boardlogger.add_scalars('F1-score progression', {'val': res['f1_score_prog']}, kvs['cur_epoch'])
    boardlogger.add_scalars('Average Precision progression', {'val': res['ap_prog']}, kvs['cur_epoch'])

    kvs.update(f'losses_fold_[{kvs["cur_fold"]}]', {'epoch': kvs['cur_epoch'],
                                                    'train_loss': train_loss,
                                                    'val_loss': val_loss})

    kvs.update(f'val_metrics_fold_[{kvs["cur_fold"]}]', res)

    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))
