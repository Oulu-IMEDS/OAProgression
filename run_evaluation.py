import os
import gc
import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import cv2


from oaprogression.evaluation import tools, stats, gcam


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--from_cache', default=False)
    parser.add_argument('--n_bootstrap', type=int, default=10000)
    parser.add_argument('--snapshot', default='2018_11_12_16_15')
    parser.add_argument('--save_dir', default='')
    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_snapshot = pickle.load(f)

    loader = tools.init_loader(session_snapshot['metadata_test'][0], args)

    gradcam_maps_all = 0
    res = 0
    if not args.from_cache:
        for fold_id in range(session_snapshot['args'][0].n_folds):
            features, fc = tools.init_fold(fold_id, session_snapshot, args)

            preds = []
            gradcam_maps_fold = []
            ids = []
            sides = []
            for batch_id, sample in enumerate(tqdm(loader, total=len(loader), desc='Prediction from fold {}'.format(fold_id))):
                gcam_batch, probs_not_summed = gcam.eval_batch(sample, features, fc)
                gradcam_maps_fold.append(gcam_batch)
                preds.append(probs_not_summed)
                ids.extend(sample['ID_SIDE'])
                gc.collect()

            preds = np.vstack(preds)
            gradcam_maps_all += np.vstack(gradcam_maps_fold)
            res += preds
            gc.collect()

        res /= 5.
        np.savez_compressed(os.path.join(args.save_dir, 'results.npz'),
                            gradcam_maps_all=gradcam_maps_all,
                            preds=res,
                            ids=ids)

    data = np.load(os.path.join(args.save_dir, 'results.npz'))

    gcams = data['gradcam_maps_all']
    preds = data['preds'][:, 1:].sum(1)
    ids = data['ids']

    res = pd.DataFrame(data={'ID': list(map(lambda x: x.split('_')[0], ids)),
                             'Side': list(map(lambda x: x.split('_')[1], ids)), 'pred': preds})

    res = pd.merge(session_snapshot['metadata_test'][0], res, on=('ID', 'Side'))
    res.Progressor = res.Progressor > 0

    print('# subjects', np.unique(res.ID).shape[0])
    print('# knees', res.shape[0])
    print('# progressors', res[res.Progressor == 1].shape[0])
    print('# non-progressors', res[res.Progressor == 0].shape[0])
    print('')
    print('All knees:')
    print('-------------')

    stats.roc_curve_bootstrap(res.Progressor.values.flatten(),
                              res.pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'auc_MOST_DL.pdf'))
    print('')
    print('KL0 at baseline:')
    print('----------------')

    stats.roc_curve_bootstrap(res[res.KL == 0].Progressor.values.flatten(),
                              res[res.KL == 0].pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'auc_MOST_DL_0.pdf'))
    print('')
    print('KL1 at baseline:')
    print('----------------')

    stats.roc_curve_bootstrap(res[res.KL == 1].Progressor.values.flatten(),
                              res[res.KL == 1].pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'auc_MOST_DL_0.pdf'))

    gcam.preds_and_hmaps(rs_result=res[(res.KL == 0) | (res.KL == 1)],
                         gradcams=gcams,
                         dataset_root=args.dataset_root,
                         figsize=16,
                         threshold=0.8,
                         savepath=args.save_dir)









