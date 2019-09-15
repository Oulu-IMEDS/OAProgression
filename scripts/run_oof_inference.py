import argparse
import gc
import os
import pickle
import pprint

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from oaprogression.evaluation import tools

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='')
    parser.add_argument('--metadata_root', default='')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--from_cache', default=False)
    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_snapshot = pickle.load(f)
    preds_prog = []
    preds_kl = []

    ids = []
    if not args.from_cache:
        for fold_id in range(session_snapshot['args'][0].n_folds):
            features, fc_prog, fc_kl = tools.init_fold(fold_id, session_snapshot, os.path.join(args.snapshots_root,
                                                                                               args.snapshot),
                                                       return_fc_kl=True)

            _, val_index = session_snapshot['cv_split_train'][0][fold_id]
            x_val = session_snapshot['metadata'][0].iloc[val_index]
            loader = tools.init_loader(x_val, args, args.snapshots_root)

            for batch_id, sample in enumerate(tqdm(loader, total=len(loader),
                                                   desc='Prediction from fold {}'.format(fold_id))):
                probs_prog_batch, probs_kl_batch = tools.eval_batch(sample, features, fc_prog, fc_kl)
                preds_prog.append(probs_prog_batch)
                preds_kl.append(probs_kl_batch)

                ids.extend(sample['ID_SIDE'])
                gc.collect()

        preds_prog = np.vstack(preds_prog)
        preds_kl = np.vstack(preds_kl)

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

        res.to_pickle(os.path.join(args.save_dir, 'oof_results.pkl'))
    else:
        res = pd.read_pickle(os.path.join(args.save_dir, 'oof_results.pkl'))

    metadata = session_snapshot['metadata'][0]
    metadata.ID = metadata.ID.astype(str)
    res.ID = res.astype(str)

    res = pd.merge(res, session_snapshot['metadata'][0], on=('ID', 'Side'))
    val_metrics = tools.calc_metrics(res.Progressor.values, res.KL.values,
                                     res[['prog_pred_0', 'prog_pred_1', 'prog_pred_2']].values,
                                     res[['kl_pred_0', 'kl_pred_1', 'kl_pred_2', 'kl_pred_3']].values, )

    pprint.pprint(val_metrics)
