import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from oaprogression.evaluation import tools
from oaprogression.training.lgbm_tools import optimize_lgbm_hyperopt, fit_lgb
from oaprogression.training.stacking import init_first_level_data_for_stacking
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

    train_folds, metadata_test = init_first_level_data_for_stacking(args, session_snapshot)

    seed = session_snapshot['args'][0].seed
    results = {}
    for feature_set in [['AGE', 'SEX', 'BMI', 'SURG', 'INJ', 'WOMAC', 'kl_pred_0', 'kl_pred_1', 'kl_pred_2',
                         'kl_pred_3', 'prog_pred_0', 'prog_pred_1', 'prog_pred_2'],
                        ['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC', 'kl_pred_0', 'kl_pred_1', 'kl_pred_2',
                         'kl_pred_3', 'prog_pred_0', 'prog_pred_1', 'prog_pred_2']]:
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
        results[f'preds_MOST_BL_all_{features_suffix}'] = (ids, sides, y_test, test_res)
        results[f'lgbm_params_oof_preds_OAI_BL_all_{features_suffix}'] = (oof_preds, best_params)
        results[f'lgbm_fi_{features_suffix}'] = [gbm.feature_importance() for gbm in models_best]

    with open(os.path.join(args.save_dir, 'results_lgbm_stacking.pkl'), 'wb') as f:
        pickle.dump(results, f)
