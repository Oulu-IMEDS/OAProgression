import os
import pickle
import gc
import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

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
    res['Progressor_type'] = res.Progressor.values.copy()
    res.Progressor = res.Progressor > 0

    # Reading the clinical data
    clinical_most = pd.read_csv(os.path.join(args.metadata_root, 'MOST_participants.csv'))
    res = pd.merge(res, clinical_most, on='ID')

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

    print('All knees (young):')
    print('-------------')
    
    plt.rcParams.update({'font.size': 16})
    stats.roc_curve_bootstrap(res[res.AGE < 60].Progressor.values.flatten(),
                              res[res.AGE < 60].pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'ROC_MOST_DL_young.pdf'))


    print('')
    print('KL0 at baseline:')
    print('----------------')
    plt.rcParams.update({'font.size': 16})
    stats.roc_curve_bootstrap(res[res.KL == 0].Progressor.values.flatten(),
                              res[res.KL == 0].pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'ROC_MOST_DL_0.pdf'))
    print('')
    print('KL0 at baseline (young):')
    print('----------------')
    plt.rcParams.update({'font.size': 16})
    stats.roc_curve_bootstrap(res[(res.AGE < 60) & (res.KL == 0)].Progressor.values.flatten(),
                              res[(res.AGE < 60) & (res.KL == 0)].pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'ROC_MOST_DL_0_young.pdf'))

    print('')
    print('KL1 at baseline:')
    print('----------------')
    plt.rcParams.update({'font.size': 16})
    stats.roc_curve_bootstrap(res[res.KL == 1].Progressor.values.flatten(),
                              res[res.KL == 1].pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'ROC_MOST_DL_1.pdf'))

    print('')
    print('KL0-1 at baseline (no OA):')
    print('----------------')
    plt.rcParams.update({'font.size': 16})
    stats.roc_curve_bootstrap(res[res.KL < 2].Progressor.values.flatten(),
                              res[res.KL < 2].pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'ROC_MOST_DL_01.pdf'))

    print('')
    print('KL0-1 at baseline (no OA, young):')
    print('----------------')
    plt.rcParams.update({'font.size': 16})
    stats.roc_curve_bootstrap(res[(res.KL < 2) & (res.AGE < 60)].Progressor.values.flatten(),
                              res[(res.KL < 2) & (res.AGE < 60)].pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'ROC_MOST_DL_01_young.pdf'))

    
    print('')
    print('KL2-4 at baseline (OA):')
    print('----------------')
    plt.rcParams.update({'font.size': 16})
    stats.roc_curve_bootstrap(res[res.KL >= 2].Progressor.values.flatten(),
                              res[res.KL >= 2].pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'ROC_MOST_DL_234.pdf'))

    print('')
    print('KL2-4 at baseline (OA, young):')
    print('----------------')
    plt.rcParams.update({'font.size': 16})
    stats.roc_curve_bootstrap(res[(res.KL >= 2) & (res.AGE < 60)].Progressor.values.flatten(),
                              res[(res.KL >= 2) & (res.AGE < 60)].pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'ROC_MOST_DL_234_young.pdf'))


    print('')
    print('KL2 at baseline:')
    print('----------------')
    plt.rcParams.update({'font.size': 16})
    stats.roc_curve_bootstrap(res[res.KL == 2].Progressor.values.flatten(),
                              res[res.KL == 2].pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'ROC_MOST_DL_2.pdf'))

    print('')
    print('KL3 at baseline:')
    print('----------------')
    plt.rcParams.update({'font.size': 16})
    stats.roc_curve_bootstrap(res[res.KL == 3].Progressor.values.flatten(),
                              res[res.KL == 3].pred.values.flatten(),
                              n_bootstrap=args.n_bootstrap,
                              savepath=os.path.join(args.save_dir, f'ROC_MOST_DL_3.pdf'))

    with open(os.path.join(args.save_dir, 'results_baselines.pkl'), 'rb') as f:
         baseline_results = pickle.load(f)

    bl_ids, bl_sides, _, bl_pred = baseline_results['preds_MOST_BL_all_AGE_SEX_BMI_KL']
    bl_df = pd.DataFrame(data={'ID': bl_ids, 'Side': bl_sides, 'bl_pred': bl_pred})

    merged_bl_dl = pd.merge(res, bl_df, on=('ID', 'Side'))

    logp = stats.delong_roc_test(merged_bl_dl.Progressor.values.astype(float).flatten(), \
                                 merged_bl_dl.pred.values.flatten(), \
                                 merged_bl_dl.bl_pred.values.flatten())
    print('P-value (DeLong DL vs Baseline:', 10**logp)

    stats.compare_curves(merged_bl_dl.Progressor.values.astype(float).flatten(), \
                         merged_bl_dl.bl_pred.values.flatten(), \
                         merged_bl_dl.pred.values.flatten(), \
                         savepath_roc=os.path.join(args.save_dir, 'ROC_MOST_DL_bl_superimposed.pdf'),
                         savepath_pr=os.path.join(args.save_dir, 'PR_MOST_DL_bl_superimposed.pdf'))
    

    if args.plot_gcams:
        gcam.preds_and_hmaps(rs_result=res,
                             gradcams=gcams,
                             dataset_root=args.dataset_root,
                             figsize=10,
                             threshold=0.3,
                             savepath=args.save_dir)
    








