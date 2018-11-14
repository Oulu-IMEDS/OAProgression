from oaprogression.training import baselines
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import numpy as np

if __name__ == "__main__":
    args = baselines.init_args()
    train_folds, metadata_test = baselines.init_metadata(args)

    c_vals = np.logspace(-6, 0, 1000)
    for feature_set in [['AGE', 'SEX', 'BMI'], ['AGE', 'SEX', 'BMI', 'KL']]:
        cv_scores = []
        for C in c_vals:
            folds_predicts = []
            folds_gt = []
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

                clf = LogisticRegression(C=C, random_state=args.seed)
                clf.fit(X_train, y_train)
                p_val = clf.predict_proba(X_val)[:, 1]

                folds_predicts.extend(p_val.flatten().tolist())
                folds_gt.extend(y_val.flatten().tolist())

            auc = roc_auc_score(folds_gt, folds_predicts)
            cv_scores.append(auc)

        opt_c_id = np.argmax(cv_scores)

        print(feature_set, c_vals[opt_c_id], cv_scores[opt_c_id])






