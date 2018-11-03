import os
import gc
import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import cv2


from oaprogression.evaluation import rstools, stats, gcam


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/data/DL_spring2/OA_progression_project/Data/RS_data/')
    parser.add_argument('--rs_cohort', default=3)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--snapshots_root', default='/data/DL_spring2/OA_progression_project/snapshots')
    parser.add_argument('--from_cache', default=False)
    parser.add_argument('--n_bootstrap', type=int, default=10000)
    parser.add_argument('--snapshot', default='2018_11_03_10_38')
    parser.add_argument('--save_dir', default='/data/DL_spring2/OA_progression_project/Results')
    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_snapshot = pickle.load(f)

    rs_meta = pd.read_csv(os.path.join(args.data_root, 'RS_metadata.csv'))
    rs_meta_preselected = pd.read_csv(os.path.join(args.data_root, f'RS{args.rs_cohort}', 'RS3_preselected.csv'))
    rs_meta = rstools.preprocess_rs_meta(rs_meta, rs_meta_preselected, 3)

    loader = rstools.init_loader(rs_meta, args)

    gradcam_maps_all = 0
    res = 0
    if not args.from_cache:
        for fold_id in range(session_snapshot['args'][0].n_folds):
            features, fc = rstools.init_fold(fold_id, session_snapshot, args)

            preds = []
            gradcam_maps_fold = []
            ids = []
            sides = []
            for batch_id, sample in enumerate(tqdm(loader, total=len(loader), desc='Prediction from fold {}'.format(fold_id))):
                gcam_batch, probs_not_summed = gcam.eval_batch(sample, features, fc)
                gradcam_maps_fold.append(gcam_batch)
                preds.append(probs_not_summed)
                ids.extend(sample['ergoid'].numpy().tolist())
                sides.extend(sample['side'])
                gc.collect()

            preds = np.vstack(preds)
            gradcam_maps_all += np.vstack(gradcam_maps_fold)
            res += preds
            gc.collect()

        res /= 5.
        np.savez_compressed(os.path.join(args.save_dir, f'RS{args.rs_cohort}.npz'),
                            gradcam_maps_all=gradcam_maps_all,
                            preds=res,
                            ids=ids,
                            sides=sides)

    data = np.load(os.path.join(args.save_dir, f'RS{args.rs_cohort}.npz'))

    gcams = data['gradcam_maps_all']
    preds = data['preds'][:, 1:].sum(1)
    ergoids = data['ids']
    sides = data['sides']

    res = pd.DataFrame(data={'ergoid': ergoids, 'side': sides, 'pred': preds})

    res = pd.merge(rs_meta, res, on=('ergoid', 'side'))
    """
    print('# subjects', np.unique(res.ergoid).shape[0])
    print('# knees', res.shape[0])
    print('# progressors', res[res.progressor == 1].shape[0])
    print('# non-progressors', res[res.progressor == 0].shape[0])
    print('')
    print('All knees:')
    print('-------------')
    stats.roc_curve_bootstrap(res.progressor.values.flatten(),
                              res.pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'auc_all_subjects_RS{args.rs_cohort}.pdf'))
    print('')
    print('KL0 at baseline:')
    print('----------------')

    stats.roc_curve_bootstrap(res[res.kl1 == 0].progressor.values.flatten(),
                              res[res.kl1 == 0].pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'auc_kl0_bl_RS{args.rs_cohort}.pdf'))
    print('')
    print('KL1 at baseline:')
    print('----------------')

    stats.roc_curve_bootstrap(res[res.kl1 == 1].progressor.values.flatten(),
                              res[res.kl1 == 1].pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'auc_kl1_bl_RS{args.rs_cohort}.pdf'))
    """
    gcam.preds_and_hmaps(rs_result=res[(res.kl1 == 0) | (res.kl1 == 1)],
                         gradcams=gcams,
                         dataset_root=os.path.join(args.data_root, f'RS{args.rs_cohort}', 'localized'),
                         figsize=16,
                         threshold=0.5,
                         savepath=args.save_dir)









