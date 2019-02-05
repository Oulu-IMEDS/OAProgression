import os

if int(os.getenv('USE_AGG', 1)) == 1:
    import matplotlib

    matplotlib.use('Agg')

import pickle
from sklearn.metrics import average_precision_score, roc_auc_score

from oaprogression.training import baselines
from oaprogression.evaluation import tools

if __name__ == "__main__":
    args = baselines.init_args()
    train_folds, metadata_test, seed = baselines.init_metadata(args)
    os.makedirs(args.save_dir, exist_ok=True)

    results = {}
    for feature_set in [['AGE', 'SEX', 'BMI'],
                        ['KL', ],
                        ['AGE', 'SEX', 'BMI', 'SURG', 'INJ', 'WOMAC'],
                        ['AGE', 'SEX', 'BMI', 'KL'],
                        ['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC']]:

        for regularize, model_type in zip([False, True], ['statsmodels', 'sklearn']):
            models_best, mean_std_best, folds_gt, folds_preds = baselines.build_logreg_model(train_folds, feature_set,
                                                                                             seed,
                                                                                             args.n_vals_c,
                                                                                             average_precision_score,
                                                                                             regularize)

            test_res = tools.eval_models(metadata_test, feature_set, models_best,
                                         mean_std_best, model_type=model_type)

            features_suffix = '_'.join(feature_set)

            y_test = metadata_test.Progressor.values.copy() > 0
            ids = metadata_test.ID.values
            sides = metadata_test.Side.values

            print(f'[{model_type} | {feature_set}] CV AUC {roc_auc_score(folds_gt, folds_preds):.5} |'
                  f' AP {average_precision_score(folds_gt, folds_preds):.5}')

            print(f'[{model_type} | {feature_set}] Test AUC {roc_auc_score(y_test, test_res):.5} |'
                  f' AP {average_precision_score(y_test, test_res):.5}')

            if model_type != "sklearn":
                results[f'preds_MOST_BL_all_{features_suffix}'] = (ids, sides, y_test, test_res)
        print('=' * 80)

    with open(os.path.join(args.save_dir, 'results_baselines_logreg.pkl'), 'wb') as f:
        pickle.dump(results, f)
