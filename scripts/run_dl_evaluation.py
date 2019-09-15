import argparse
import gc
import os
import pickle

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from oaprogression.evaluation import tools, gcam

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

    os.makedirs(args.save_dir, exist_ok=True)

    if not args.from_cache:
        loader = tools.init_loader(pd.read_csv(os.path.join(args.metadata_root, 'MOST_progression.csv')), args,
                                   args.snapshots_root)
        tools.run_test_inference(loader, session_snapshot, args.snapshots_root, args.snapshot, args.save_dir)

    data = np.load(os.path.join(args.save_dir, 'results.npz'))

    gcams = data['gradcam_maps_all']
    preds = data['preds_prog'][:, 1:].sum(1)
    ids = data['ids']

    res = pd.DataFrame(data={'ID': list(map(lambda x: x.split('_')[0], ids)),
                             'Side': list(map(lambda x: x.split('_')[1], ids)), 'pred': preds})

    res = pd.merge(session_snapshot['metadata_test'][0], res, on=('ID', 'Side'))
    res['Progressor_type'] = res.Progressor.values.copy()
    res.Progressor = res.Progressor > 0

    if args.plot_gcams:
        os.makedirs(os.path.join(args.save_dir, 'prog_heatmaps'), exist_ok=True)
        gcam.preds_and_hmaps(results=res,
                             gradcams=gcams,
                             dataset_root=args.dataset_root,
                             figsize=10,
                             threshold=0.4,
                             savepath=os.path.join(args.save_dir, 'prog_heatmaps'),
                             gcam_type='prog')
        os.makedirs(os.path.join(args.save_dir, 'nonprog_heatmaps'), exist_ok=True)
        gcam.preds_and_hmaps(results=res,
                             gradcams=gcams,
                             dataset_root=args.dataset_root,
                             figsize=10,
                             threshold=0.4,
                             savepath=os.path.join(args.save_dir, 'nonprog_heatmaps'),
                             gcam_type='non-prog')
