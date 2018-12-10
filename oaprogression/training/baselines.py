import pickle
import argparse
import os
import pandas as pd
import numpy as np
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


def build_logreg_model(train_folds, feature_set, seed, n_vals_c, metric):
    cv_scores = []
    models = []
    means_stds = []
    c_vals = np.logspace(-6, 2, n_vals_c)
    for C in c_vals:  # Enumerating the regularizer weight
        folds_predicts = []
        folds_gt = []
        folds_models = []
        folds_means_stds = []
        for fold_id, (train_split, val_split) in enumerate(train_folds):  # Going through the prepared fold splits
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

            clf = LogisticRegression(C=C, random_state=seed, solver='lbfgs')
            clf.fit(X_train, y_train)
            p_val = clf.predict_proba(X_val)[:, 1]

            folds_predicts.extend(p_val.flatten().tolist())
            folds_gt.extend(y_val.flatten().tolist())
            folds_models.append(clf)
            folds_means_stds.append([mean, std])

        auc = metric(folds_gt, folds_predicts)
        cv_scores.append(auc)
        models.append(folds_models)
        means_stds.append(folds_means_stds)

    opt_c_id = np.argmax(cv_scores)
    models_best = models[opt_c_id]
    mean_std_best = means_stds[opt_c_id]
    return models_best, mean_std_best, cv_scores[opt_c_id]

