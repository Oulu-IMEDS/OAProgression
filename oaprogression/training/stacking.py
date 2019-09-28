import os
import numpy as np
import pandas as pd
from oaprogression.training.baselines import init_metadata_test


def init_first_level_data_for_stacking(args, session_snapshot):
    oof_preds = pd.read_pickle(os.path.join(args.save_dir, 'oof_results.pkl'))
    oof_preds.ID = oof_preds.ID.values.astype(int)
    metadata_clinical = pd.read_csv(os.path.join(args.metadata_root, 'OAI_participants.csv'))
    metadata = session_snapshot['metadata'][0]

    metadata = pd.merge(metadata, metadata_clinical, on=('ID', 'Side'))
    metadata = pd.merge(metadata, oof_preds, on=('ID', 'Side'))
    metadata.SEX = metadata.SEX.values.astype(int)
    metadata.INJ = metadata.INJ.values.astype(int)
    metadata.SURG = metadata.SURG.values.astype(int)
    metadata['prog_pred'] = metadata.prog_pred_1 + metadata.prog_pred_2

    train_folds = []
    for train_idx, val_idx in session_snapshot['cv_split_all_folds'][0]:
        train_folds.append((metadata.iloc[train_idx], metadata.iloc[val_idx]))

    data = np.load(os.path.join(args.save_dir, 'results.npz'))

    preds_prog = data['preds_prog']
    preds_kl = data['preds_kl']
    ids = data['ids']
    res = pd.DataFrame(data={'ID': list(map(lambda x: x.split('_')[0], ids)),
                             'Side': list(map(lambda x: x.split('_')[1], ids)),
                             'prog_pred_0': preds_prog[:, 0],
                             'prog_pred_1': preds_prog[:, 1],
                             'prog_pred_2': preds_prog[:, 2],
                             'kl_pred_0': preds_kl[:, 0],
                             'kl_pred_1': preds_kl[:, 1],
                             'kl_pred_2': preds_kl[:, 2],
                             'kl_pred_3': preds_kl[:, 3],
                             })

    metadata_test = pd.merge(res, init_metadata_test(args), on=('ID', 'Side'))
    metadata_test.Progressor = metadata_test.Progressor > 0
    metadata_test['prog_pred'] = metadata_test.prog_pred_2 + metadata_test.prog_pred_1

    return train_folds, metadata_test