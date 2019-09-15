import sys
import os
import cv2
import argparse
import pickle

from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from oaprogression.metadata.oai import jsw_features, read_jsw_metadata_oai, beam_angle_feature
from oaprogression.training.lgbm_tools import optimize_lgbm_hyperopt, fit_lgb
from oaprogression.evaluation import tools


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

DEBUG = sys.gettrace() is not None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='')
    parser.add_argument('--metadata_root', default='')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--lgbm_hyperopt_trials', type=int, default=2)
    parser.add_argument('--save_dir', default='')
    args = parser.parse_args()

    sites, metadata = read_jsw_metadata_oai(args.metadata_root, args.dataset_root)

    results = {}
    for feature_set in [['AGE', 'SEX', 'BMI'],
                        ['AGE', 'SEX', 'BMI', 'SURG', 'INJ', 'WOMAC'],
                        ['AGE', 'SEX', 'BMI', 'KL'],
                        ['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC'],
                        ['AGE', 'SEX', 'BMI', beam_angle_feature], # Reproducing the test results w. beam angle
                        ['AGE', 'SEX', 'BMI', 'SURG', 'INJ', 'WOMAC', beam_angle_feature],
                        ['AGE', 'SEX', 'BMI', 'KL', beam_angle_feature],
                        ['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC', beam_angle_feature],
                        ['AGE', 'SEX', 'BMI'] + jsw_features, # Adding JSW to the base model
                        ['AGE', 'SEX', 'BMI', 'KL'] + jsw_features,
                        ['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC'] + jsw_features,
                        ['AGE', 'SEX', 'BMI', 'SURG', 'INJ', 'WOMAC'] + jsw_features,
                        ['AGE', 'SEX', 'BMI', beam_angle_feature] + jsw_features, # Let's try to add the beam angle as well
                        ['AGE', 'SEX', 'BMI', 'KL', beam_angle_feature] + jsw_features,
                        ['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC', beam_angle_feature] + jsw_features,
                        ['AGE', 'SEX', 'BMI', 'SURG', 'INJ', 'WOMAC', beam_angle_feature] + jsw_features,
                        ]:

        features_suffix = '_'.join(feature_set)
        results[features_suffix] = {}
        for test_site in sites:
            top_subj_train = metadata[metadata.V00SITE != test_site]
            top_subj_test = metadata[metadata.V00SITE == test_site]

            gkf = GroupKFold(n_splits=5)
            train_folds = []
            for train_idx, val_idx in gkf.split(top_subj_train, y=top_subj_train.Progressor, groups=top_subj_train.ID):
                train_folds.append((top_subj_train.iloc[train_idx], top_subj_train.iloc[val_idx]))

            best_params, trials = optimize_lgbm_hyperopt(train_folds, feature_set,
                                                         average_precision_score,
                                                         args.seed, hyperopt_trials=args.lgbm_hyperopt_trials)

            ap_score, models_best, oof_preds = fit_lgb(best_params, train_folds,
                                                       feature_set, average_precision_score, True, True)

            print('CV score:', feature_set, ap_score)
            test_res = tools.eval_models(top_subj_test, feature_set, models_best, mean_std_best=None,
                                         impute=False, model_type='lgbm')

            y_test = top_subj_test.Progressor.values.copy() > 0
            ids = top_subj_test.ID.values
            sides = top_subj_test.Side.values

            results[features_suffix][test_site] = (ids, sides, y_test, test_res)

    with open(os.path.join(args.save_dir, 'results_ablation_jsw_lgbm.pkl'), 'wb') as f:
        pickle.dump(results, f)
