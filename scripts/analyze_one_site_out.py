import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from oaprogression.evaluation import tools
from oaprogression.training.lgbm_tools import optimize_lgbm_hyperopt, fit_lgb
from oaprogression.metadata.oai import jsw_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots_dir', default='')
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--lgbm_hyperopt_trials', type=int, default=500)
    parser.add_argument('--jsw_ablation_path', default='')
    parser.add_argument('--save_dir', default='')
    args = parser.parse_args()

    results = {}
    for site_folder in glob.glob(os.path.join(args.snapshots_dir, args.snapshot, 'site_*/*/')):
        tmp = {}
        # Preparing the OOF preds for the ensemble
        with open(os.path.join(os.path.split(os.path.split(site_folder)[0])[0], 'session.pkl'), 'rb') as f:
            session_snapshot = pickle.load(f)

        res_oof = pd.read_pickle(os.path.join(site_folder, 'oof_results.pkl'))
        res_oof.ID = res_oof.ID.astype(int)

        metadata = pd.merge(session_snapshot['metadata'][0], res_oof, on=('ID', 'Side'))

        train_folds = []
        for train_idx, val_idx in session_snapshot['cv_split_all_folds'][0]:
            train_folds.append((metadata.iloc[train_idx], metadata.iloc[val_idx]))

        # Preparing the test predictions
        data = np.load(os.path.join(site_folder, 'results.npz'))
        preds_prog = data['preds_prog']
        preds_kl = data['preds_kl']
        ids = data['ids']
        dl_preds = pd.DataFrame(data={'ID': list(map(lambda x: int(x.split('_')[0]), ids)),
                                      'Side': list(map(lambda x: x.split('_')[1], ids)),
                                      'Prediction': preds_prog[:, 1:].sum(1),
                                      'prog_pred_0': preds_prog[:, 0],
                                      'prog_pred_1': preds_prog[:, 1],
                                      'prog_pred_2': preds_prog[:, 2],
                                      'kl_pred_0': preds_kl[:, 0],
                                      'kl_pred_1': preds_kl[:, 1],
                                      'kl_pred_2': preds_kl[:, 2],
                                      'kl_pred_3': preds_kl[:, 3],
                                      })

        with open(os.path.join(os.path.split(os.path.split(site_folder)[0])[0], 'session.pkl'), 'rb') as f:
            session = pickle.load(f)

        metadata_test = pd.merge(session['metadata_test'][0], dl_preds, on=('ID', 'Side'))

        tmp['cnn'] = (metadata_test.ID.values, metadata_test.Side.values,
                      metadata_test.Progressor.values > 0,
                      metadata_test.Prediction.values)

        for feature_set_id, feature_set in enumerate([['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC',
                                                       'kl_pred_0', 'kl_pred_1', 'kl_pred_2',
                                                       'kl_pred_3',
                                                       'prog_pred_0', 'prog_pred_1', 'prog_pred_2'],
                                                      ['AGE', 'SEX', 'BMI', 'SURG', 'INJ', 'WOMAC',
                                                       'kl_pred_0', 'kl_pred_1', 'kl_pred_2', 'kl_pred_3',
                                                       'prog_pred_0', 'prog_pred_1', 'prog_pred_2']]):

            best_params, trials = optimize_lgbm_hyperopt(train_folds, feature_set,
                                                         average_precision_score,
                                                         session_snapshot['args'][0].seed, args.lgbm_hyperopt_trials)

            ap_score, models_best, oof_preds = fit_lgb(best_params, train_folds,
                                                       feature_set, average_precision_score, True, True)

            lgbm_test_res = tools.eval_models(metadata_test, feature_set, models_best, mean_std_best=None,
                                              impute=False, model_type='lgbm')

            tmp[f'lgbm_{feature_set_id}'] = (metadata_test.ID.values, metadata_test.Side.values,
                                             metadata_test.Progressor.values > 0,
                                             lgbm_test_res)

        results[site_folder.split("/")[-3]] = tmp

    ap = {model: list() for model in results['site_A']}
    auc = {model: list() for model in results['site_A']}
    for site in results.keys():
        for model in auc:
            ids, sides, y_test, test_res = results[site][model]
            ap[model].append(average_precision_score(y_test, test_res))
            auc[model].append(roc_auc_score(y_test, test_res))

    for model in auc:
        print('================')
        print(f'{model} AP: {np.mean(ap[model]):.2}±{np.std(ap[model]):.2}')
        print(f'{model} AUC: {np.mean(auc[model]):.2}±{np.std(auc[model]):.2}')

    with open(os.path.join(args.save_dir, 'results_one_site_out_all.pkl'), 'wb') as f:
        pickle.dump(results, f)

    with open(args.jsw_ablation_path, 'rb') as f:
        data = pickle.load(f)

    for features in data.keys():
        model = features.replace('_'.join(jsw_features), 'JSWs').replace('V00BMANG', 'BeamAngle')
        ap = []
        auc = []
        for site in data[features]:
            ids, sides, y_test, test_res = data[features][site]
            ap.append(average_precision_score(y_test, test_res))
            auc.append(roc_auc_score(y_test, test_res))
        print('================')
        print(f'{model} AP: {np.mean(ap):.2}±{np.std(ap):.2}')
        print(f'{model} AUC: {np.mean(auc):.2}±{np.std(auc):.2}')
