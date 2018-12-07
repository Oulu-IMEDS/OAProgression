import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from oaprogression.training import baselines
from oaprogression.evaluation import stats

if __name__ == "__main__":
    args = baselines.init_args()
    train_folds, metadata_test = baselines.init_metadata(args)
    os.makedirs(args.save_dir, exist_ok=True)

    c_vals = np.logspace(-6, 0, args.n_vals_c)
    results = {}
    for feature_set in [['AGE', ],
                        ['SEX', ],
                        ['BMI', ],
                        ['KL', ],
                        ['AGE', 'SEX', 'BMI', 'KL'],
                        ['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC']]:
        cv_scores = []
        models = []
        means_stds = []
        for C in c_vals:
            folds_predicts = []
            folds_gt = []
            folds_models = []
            folds_means_stds = []
            for fold_id, (train_split, val_split) in enumerate(train_folds):
                X_train = train_split[feature_set].values.astype(float)
                X_val = val_split[feature_set].values.astype(float)

                y_train = train_split.Progressor.values > 0
                y_val = val_split.Progressor.values > 0

                y_train = y_train[~np.isnan(X_train).any(1)]
                X_train = X_train[~np.isnan(X_train).any(1)]

                y_val = y_val[~np.isnan(X_val).any(1)]
                X_val = X_val[~np.isnan(X_val).any(1)]

                mean = np.mean(X_train, 0)
                std = np.std(X_train, 0)

                X_train -= mean
                X_train /= std

                X_val -= mean
                X_val /= std

                clf = LogisticRegression(C=C, random_state=args.seed, solver='lbfgs')
                clf.fit(X_train, y_train)
                p_val = clf.predict_proba(X_val)[:, 1]

                folds_predicts.extend(p_val.flatten().tolist())
                folds_gt.extend(y_val.flatten().tolist())
                folds_models.append(clf)
                folds_means_stds.append([mean, std])

            auc = roc_auc_score(folds_gt, folds_predicts)
            cv_scores.append(auc)
            models.append(folds_models)
            means_stds.append(folds_means_stds)

        opt_c_id = np.argmax(cv_scores)
        models_best = models[opt_c_id]
        mean_std_best = means_stds[opt_c_id]

        print('CV score:', feature_set, c_vals[opt_c_id], cv_scores[opt_c_id])

        for kl_bl in [-2, -1, 0, 1]:

            if kl_bl == -1:
                 X_test_initial = metadata_test[metadata_test.KL < 2]
                 kl_bl='0_1'
            elif kl_bl == -2:
                kl_bl = 'all'
                X_test_initial = metadata_test
            else:
                X_test_initial = metadata_test[metadata_test.KL == kl_bl]

            print(f'Baseline KL: [{kl_bl}]')
            y_test = X_test_initial.Progressor.values.copy() > 0
            ids = X_test_initial.ID.values
            sides = X_test_initial.Side.values
            # Using mean imputation for logreg
            X_test_initial.fillna(X_test_initial.mean(), inplace=True)
            X_test_initial = X_test_initial[feature_set].values.astype(float).copy()

            test_res = 0
            for model_id in range(len(models_best)):
                mean, std = mean_std_best[model_id]
                X_test = X_test_initial.copy()
                X_test -= mean
                X_test /= std

                test_res += models_best[model_id].predict_proba(X_test)[:, 1]

            test_res /= len(models_best)

            features_suffix = '_'.join(feature_set)
            plt.rcParams.update({'font.size': 16})
            stats.roc_curve_bootstrap(y_test,
                                      test_res,
                                      n_bootstrap=args.n_bootstrap,
                                      savepath=os.path.join(args.save_dir,
                                                            f'ROC_MOST_BL_{kl_bl}_{features_suffix}.pdf'))

            results[f'preds_MOST_BL_{kl_bl}_{features_suffix}'] = (ids, sides, y_test, test_res)

    with open(os.path.join(args.save_dir, 'results_baselines.pkl'), 'wb') as f:
        pickle.dump(results, f)




