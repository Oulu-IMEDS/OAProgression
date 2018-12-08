import os
import gc
import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

from oaprogression.evaluation import tools, stats

from sklearn.metrics import average_precision_score
from oaprogression.training.lgbm_tools import optimize_lgbm_hyperopt, fit_lgb
from oaprogression.training.baselines import init_metadata_test

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='')
    parser.add_argument('--metadata_root', default='')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--from_cache', default=False)
    parser.add_argument('--lgbm_hyperopt_trials', type=int, default=500)
    parser.add_argument('--n_bootstrap', type=int, default=2000)
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--plot_gcams', type=bool, default=False)
    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_snapshot = pickle.load(f)

    seed = session_snapshot['args'][0].seed
    oof_preds = pd.read_pickle(os.path.join(args.save_dir, 'oof_results.pkl'))
    oof_preds.ID = oof_preds.ID.values.astype(int)
    metadata_clinical = pd.read_csv(os.path.join(args.metadata_root, 'OAI_participants.csv'))
    metadata = session_snapshot['metadata'][0]

    metadata = pd.merge(metadata, metadata_clinical, on=('ID', 'Side'))
    metadata = pd.merge(metadata, oof_preds, on=('ID', 'Side'))
    metadata.SEX = metadata.SEX.values.astype(int)
    metadata.INJ = metadata.INJ.values.astype(int)
    metadata.SURG = metadata.SURG.values.astype(int)
    metadata['prog_pred'] = metadata.prog_pred_1 + metadata.prog_pred_2

    train_folds = []
    for train_idx, val_idx in session_snapshot['cv_split_all_folds'][0]:
        train_folds.append((metadata.iloc[train_idx], metadata.iloc[val_idx]))

    data = np.load(os.path.join(args.save_dir, 'results.npz'))

    preds_prog = data['preds_prog']
    preds_kl = data['preds_kl']
    ids = data['ids']
    res = pd.DataFrame(data={'ID': list(map(lambda x: x.split('_')[0], ids)),
                             'Side': list(map(lambda x: x.split('_')[1], ids)),
                             'prog_pred_0': preds_prog[:, 0],
                             'prog_pred_1': preds_prog[:, 1],
                             'prog_pred_2': preds_prog[:, 2],
                             'kl_pred_0': preds_kl[:, 0],
                             'kl_pred_1': preds_kl[:, 1],
                             'kl_pred_2': preds_kl[:, 2],
                             'kl_pred_3': preds_kl[:, 3],
                             'kl_pred_4': preds_kl[:, 4]
                             })

    metadata_test = pd.merge(res, init_metadata_test(args), on=('ID', 'Side'))
    metadata_test.Progressor = metadata_test.Progressor > 0
    metadata_test['prog_pred'] = metadata_test.prog_pred_2 + metadata_test.prog_pred_1

    results = {}
    for feature_set in [['AGE', 'SEX', 'BMI', 'SURG', 'INJ', 'WOMAC', 'kl_pred_0', 'kl_pred_1', 'kl_pred_2',
                         'kl_pred_3', 'kl_pred_4', 'prog_pred_0', 'prog_pred_1', 'prog_pred_2'],
                        ['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC','kl_pred_0', 'kl_pred_1', 'kl_pred_2',
                         'kl_pred_3', 'kl_pred_4', 'prog_pred_0', 'prog_pred_1', 'prog_pred_2']]:

        best_params, trials = optimize_lgbm_hyperopt(train_folds, feature_set,
                                                     average_precision_score,
                                                     seed, hyperopt_trials=args.lgbm_hyperopt_trials)

        ap_score, models_best, oof_preds = fit_lgb(best_params, train_folds,
                                                   feature_set, average_precision_score, True, True)

        print('CV score:', feature_set, ap_score)
        test_res = tools.eval_models(metadata_test, feature_set, models_best, mean_std_best=None,
                                     impute=False, model_type='lgbm')

        features_suffix = '_'.join(feature_set)
        plt.rcParams.update({'font.size': 16})

        y_test = metadata_test.Progressor.values.copy() > 0
        ids = metadata_test.ID.values
        sides = metadata_test.Side.values
        stats.roc_curve_bootstrap(y_test,
                                  test_res,
                                  n_bootstrap=args.n_bootstrap,
                                  savepath=os.path.join(args.save_dir,
                                                        f'ROC_MOST_BL_all_{features_suffix}_lgbm.pdf'))

        results[f'preds_MOST_BL_all_{features_suffix}'] = (ids, sides, y_test, test_res)

    with open(os.path.join(args.save_dir, 'results_baselines_lgbm_stacking.pkl'), 'wb') as f:
        pickle.dump(results, f)







