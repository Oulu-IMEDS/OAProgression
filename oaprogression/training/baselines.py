import pickle
import argparse
import os


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_root', default='')
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--snapshot', default='')
    args = parser.parse_args()

    return args


def init_metadata(args):
    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_snapshot = pickle.load(f)

    for train_index, val_index in session_snapshot['cv_split_all_folds'][0]:
        train_split = session_snapshot['metadata'][0].iloc[train_index]
        val_split = session_snapshot['metadata'][0].iloc[val_index]

