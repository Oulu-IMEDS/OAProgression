import os
import gc
import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

from oaprogression.evaluation import tools, stats, gcam

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='')
    parser.add_argument('--metadata_root', default='')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--from_cache', default=False)
    parser.add_argument('--n_bootstrap', type=int, default=2000)
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--plot_gcams', type=bool, default=False)
    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_snapshot = pickle.load(f)

    loader = tools.init_loader(session_snapshot['metadata_test'][0], args)

    gradcam_maps_all = 0
    res_kl = 0
    res_prog = 0
    ids = None
    if not args.from_cache:
        for fold_id in range(session_snapshot['args'][0].n_folds):
            features, fc, fc_kl = tools.init_fold(fold_id, session_snapshot, args, return_fc_kl=True)

            preds_prog_fold = []
            preds_kl_fold = []
            gradcam_maps_fold = []
            ids = []
            sides = []
            for batch_id, sample in enumerate(tqdm(loader, total=len(loader), desc='Prediction from fold {}'.format(fold_id))):
                gcam_batch, probs_prog, probs_kl = gcam.eval_batch(sample, features, fc, fc_kl)
                gradcam_maps_fold.append(gcam_batch)
                preds_prog_fold.append(probs_prog)
                preds_kl_fold.append(probs_kl)
                ids.extend(sample['ID_SIDE'])
                gc.collect()

            preds_prog_fold = np.vstack(preds_prog_fold)
            preds_kl_fold = np.vstack(preds_kl_fold)
            gradcam_maps_all += np.vstack(gradcam_maps_fold)

            res_kl += preds_kl_fold
            res_prog += preds_prog_fold
            gc.collect()

        res_kl /= 5.
        res_prog /= 5.
        np.savez_compressed(os.path.join(args.save_dir, 'results.npz'),
                            gradcam_maps_all=gradcam_maps_all,
                            preds_kl=res_kl,
                            preds_prog=res_prog,
                            ids=ids)

    data = np.load(os.path.join(args.save_dir, 'results.npz'))

    gcams = data['gradcam_maps_all']
    preds = data['preds_prog'][:, 1:].sum(1)
    ids = data['ids']

    res = pd.DataFrame(data={'ID': list(map(lambda x: x.split('_')[0], ids)),
                             'Side': list(map(lambda x: x.split('_')[1], ids)), 'pred': preds})

    res = pd.merge(session_snapshot['metadata_test'][0], res, on=('ID', 'Side'))
    res['Progressor_type'] = res.Progressor.values.copy()
    res.Progressor = res.Progressor > 0

    print('# subjects', np.unique(res.ID).shape[0])
    print('# knees', res.shape[0])
    print('# progressors', res[res.Progressor == 1].shape[0])
    print('# non-progressors', res[res.Progressor == 0].shape[0])
    print('')
    print('All knees:')
    print('-------------')

    plt.rcParams.update({'font.size': 16})
    stats.roc_curve_bootstrap(res.Progressor.values.flatten(),
                              res.pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'ROC_MOST_DL.pdf'))

    if args.plot_gcams:
        gcam.preds_and_hmaps(rs_result=res,
                             gradcams=gcams,
                             dataset_root=args.dataset_root,
                             figsize=10,
                             threshold=0.3,
                             savepath=args.save_dir)
    








