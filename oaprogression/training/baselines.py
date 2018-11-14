import pickle
import argparse
import os
import pandas as pd


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_root', default='')
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--n_bootstrap', type=int, default=10000)
    args = parser.parse_args()

    return args


def init_metadata(args):
    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_snapshot = pickle.load(f)

    clinical_data_oai = pd.read_csv(os.path.join(args.metadata_root, 'OAI_participants.csv'))
    clinical_data_oai['SEX'] = 2 - clinical_data_oai['P02SEX']
    clinical_data_oai['AGE'] = clinical_data_oai['V00AGE']
    clinical_data_oai['BMI'] = clinical_data_oai['P01BMI']

    clinical_data_most = pd.read_csv(os.path.join(args.metadata_root, 'MOST_participants.csv'))
    clinical_data_most['BMI'] = clinical_data_most['V0BMI']

    metadata = session_snapshot['metadata'][0]
    metadata = pd.merge(metadata, clinical_data_oai, on='ID')

    metadata_test = session_snapshot['metadata_test'][0]
    metadata_test = pd.merge(metadata_test, clinical_data_most, on='ID')

    train_folds  = []
    for train_index, val_index in session_snapshot['cv_split_all_folds'][0]:
        train_split = metadata.iloc[train_index]
        val_split = metadata.iloc[val_index]

        train_folds.append((train_split, val_split))

    return train_folds, metadata_test

