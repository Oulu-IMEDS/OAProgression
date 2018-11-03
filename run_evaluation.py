import glob
import os
import gc
import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import cv2



from oaprogression.evaluation import rstools


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/data/DL_spring2/OA_progression_project/Data/RS_data/')
    parser.add_argument('--rs_cohort', default=3)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--snapshots_root', default='/data/DL_spring2/OA_progression_project/snapshots')
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
    for fold_id in range(session_snapshot['args'][0].n_folds):
        features, fc = rstools.init_fold(fold_id, session_snapshot, args)

        preds = []
        gradcam_maps_fold = []
        id_side = []

        for batch_id, sample in enumerate(tqdm(loader, total=len(loader), desc='Prediction from fold {}'.format(fold_id))):
            gcam_batch, probs_not_summed = rstools.eval_batch(sample, features, fc)
            gradcam_maps_fold.append(gcam_batch)
            preds.append(probs_not_summed)
            id_side.append([sample['ergoid'], sample['side']])
            gc.collect()

        preds = np.vstack(preds)
        gradcam_maps_all += np.vstack(gradcam_maps_fold)
        res += preds
        gc.collect()


