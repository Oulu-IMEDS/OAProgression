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
    parser.add_argument('--n_bootstrap', type=int, default=2000)
    parser.add_argument('--n_vals_c', type=int, default=100)
    args = parser.parse_args()

    return args


def init_metadata(args):
    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_snapshot = pickle.load(f)

    clinical_data_oai = pd.read_csv(os.path.join(args.metadata_root, 'OAI_participants.csv'))

    clinical_data_most = pd.read_csv(os.path.join(args.metadata_root, 'MOST_participants.csv'))

    metadata = session_snapshot['metadata'][0]
    metadata = pd.merge(metadata, clinical_data_oai, on='ID')

    metadata_test = session_snapshot['metadata_test'][0]
    metadata_test = pd.merge(metadata_test, clinical_data_most, on=('ID', 'Side'))

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
    c_vals = np.logspace(-6, 0, n_vals_c)
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


def eval_logreg(metadata_test, feature_set, models_best, mean_std_best):
    X_test_initial = metadata_test[feature_set].copy()
    # Using mean imputation for logreg        
    X_test_initial.fillna(X_test_initial.mean(), inplace=True)
    X_test_initial = X_test_initial.values.astype(float)
    test_res = 0
    for model_id in range(len(models_best)):
        mean, std = mean_std_best[model_id]
        X_test = X_test_initial.copy()
        X_test -= mean
        X_test /= std
        
        test_res += models_best[model_id].predict_proba(X_test)[:, 1]
        
    test_res /= len(models_best)
    return test_res
