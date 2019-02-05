import argparse
import os
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_root', default='')
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--lgbm_hyperopt_trials', type=int, default=500)
    parser.add_argument('--n_bootstrap', type=int, default=2000)
    parser.add_argument('--n_vals_c', type=int, default=300)
    args = parser.parse_args()

    return args


def init_metadata_test(args):
    clinical_data_most = pd.read_csv(os.path.join(args.metadata_root, 'MOST_participants.csv'))
    metadata_test = pd.read_csv(os.path.join(args.metadata_root, 'MOST_progression.csv'))
    return pd.merge(metadata_test, clinical_data_most, on=('ID', 'Side'))


def init_metadata(args):
    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_snapshot = pickle.load(f)

    clinical_data_oai = pd.read_csv(os.path.join(args.metadata_root, 'OAI_participants.csv'))

    metadata = session_snapshot['metadata'][0]
    metadata = pd.merge(metadata, clinical_data_oai, on=('ID', 'Side'))

    metadata_test = init_metadata_test(args)

    train_folds = []
    for train_index, val_index in session_snapshot['cv_split_all_folds'][0]:
        train_split = metadata.iloc[train_index]
        val_split = metadata.iloc[val_index]

        train_folds.append((train_split, val_split))

    return train_folds, metadata_test, session_snapshot['args'][0].seed


def build_logreg_model(train_folds, feature_set, seed, n_vals_c, metric, regularization=False):
    cv_scores = []
    models = []
    means_stds = []
    c_vals = np.logspace(-6, 2, n_vals_c)
    if not regularization:
        c_vals = [None, ]
    for C in c_vals:  # Enumerating the regularizer weight
        folds_predicts = []
        folds_gt = []
        folds_models = []
        folds_means_stds = []

        for fold_id, (train_split, val_split) in enumerate(train_folds):  # Going through the prepared fold splits
            train_split = train_split.copy()
            val_split = train_split.copy()

            train_split.dropna(inplace=True)
            val_split.dropna(inplace=True)

            train_split.Progressor = train_split.Progressor.values > 0
            val_split.Progressor = val_split.Progressor.values > 0

            X_train = train_split[feature_set].values.astype(float)
            X_val = val_split[feature_set].values.astype(float)

            mean = np.mean(X_train, 0)
            std = np.std(X_train, 0)

            X_train -= mean
            X_train /= std

            X_val -= mean
            X_val /= std

            train_split[feature_set] = X_train
            val_split[feature_set] = X_val

            if not regularization:
                model = sm.Logit(train_split.Progressor.values, sm.add_constant(X_train))
                clf = model.fit(disp=0)
                p_val = clf.predict(sm.add_constant(X_val)).flatten().tolist()
            else:
                clf = LogisticRegression(C=C, random_state=seed, solver='lbfgs')
                clf.fit(X_train, train_split.Progressor.values)
                p_val = clf.predict_proba(X_val)[:, 1].flatten().tolist()

            folds_means_stds.append([mean, std])
            folds_predicts.extend(p_val)
            folds_gt.extend(val_split.Progressor.values.flatten().tolist())
            folds_models.append(clf)

        auc = metric(folds_gt, folds_predicts)
        cv_scores.append(auc)
        models.append(folds_models)
        means_stds.append(folds_means_stds)

    opt_c_id = np.argmax(cv_scores)
    models_best = models[opt_c_id]
    mean_std_best = means_stds[opt_c_id]
    return models_best, mean_std_best, np.array(folds_gt), np.array(folds_predicts)
