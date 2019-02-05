import os

if int(os.getenv('USE_AGG', 1)) == 1:
    import matplotlib

    matplotlib.use('Agg')

import pickle
from sklearn.metrics import average_precision_score

from oaprogression.training.lgbm_tools import optimize_lgbm_hyperopt, fit_lgb
from oaprogression.training import baselines
from oaprogression.evaluation import tools

if __name__ == "__main__":
    args = baselines.init_args()
    train_folds, metadata_test, seed = baselines.init_metadata(args)
    os.makedirs(args.save_dir, exist_ok=True)

    results = {}
    for feature_set in [['AGE', 'SEX', 'BMI'],
                        ['AGE', 'SEX', 'BMI', 'SURG', 'INJ', 'WOMAC'],
                        ['AGE', 'SEX', 'BMI', 'KL'],
                        ['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC']]:
        best_params, trials = optimize_lgbm_hyperopt(train_folds, feature_set,
                                                     average_precision_score,
                                                     seed, hyperopt_trials=args.lgbm_hyperopt_trials)

        ap_score, models_best, oof_preds = fit_lgb(best_params, train_folds,
                                                   feature_set, average_precision_score, True, True)

        print('CV score:', feature_set, ap_score)
        test_res = tools.eval_models(metadata_test, feature_set, models_best, mean_std_best=None,
                                     impute=False, model_type='lgbm')

        features_suffix = '_'.join(feature_set)

        y_test = metadata_test.Progressor.values.copy() > 0
        ids = metadata_test.ID.values
        sides = metadata_test.Side.values
        results[f'preds_MOST_BL_all_{features_suffix}'] = (ids, sides, y_test, test_res)

    with open(os.path.join(args.save_dir, 'results_baselines_lgbm.pkl'), 'wb') as f:
        pickle.dump(results, f)
