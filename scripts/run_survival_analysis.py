import os

#if int(os.getenv('USE_AGG', 1)) == 1:
#    import matplotlib
#    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from lifelines import KaplanMeierFitter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='')
    parser.add_argument('--metadata_root', default='')
    args = parser.parse_args()

    data = np.load(os.path.join(args.results_dir, 'results.npz'))

    preds_prog = data['preds_prog']
    preds_kl = data['preds_kl']
    ids = data['ids']
    dl_preds = pd.DataFrame(data={'ID': list(map(lambda x: x.split('_')[0], ids)),
                                  'Side': list(map(lambda x: x.split('_')[1], ids)),
                                  'Prediction': preds_prog[:, 1:].sum(1)})

    dl_preds = pd.merge(dl_preds, pd.read_csv(os.path.join(args.metadata_root, 'MOST_progression.csv')),
                        on=('ID', 'Side'))
    dl_preds['Progressor'] = dl_preds['Progressor'] > 0
    dl_preds = dl_preds[['ID', 'Side', 'Progressor', 'Prediction', 'Progressor_visit']]

    metadata = pd.read_csv(os.path.join(args.metadata_root, 'OAI_progression.csv'))
    metadata.ID = metadata.ID.astype(str)

    # looking for the optimal threshold
    oof_preds = pd.read_pickle(os.path.join(args.results_dir, 'oof_results.pkl'))
    oof_preds.ID = oof_preds.astype(str)

    oof_preds = pd.merge(oof_preds, metadata, on=('ID', 'Side'))

    gt = oof_preds.Progressor.values > 0
    pred = oof_preds[['prog_pred_1', 'prog_pred_2']].values.sum(1)
    f_best = 0
    f_best_ind = 0
    thresholds = np.linspace(0.01, 0.9, 1000)
    for t_ind in tqdm(range(thresholds.shape[0]), total=thresholds.shape[0]):
        score = f1_score(gt, pred > thresholds[t_ind])
        if score > f_best:
            f_best_ind = t_ind
            f_best = score

    opt_thresh = thresholds[f_best_ind]

    print(classification_report(dl_preds['Progressor'].values, dl_preds['Prediction'].values > opt_thresh))

    mapping_visits = {0: 84, 1: 15, 2: 30, 3: 60, 4: 72, 5: 84}
    groups = dl_preds['Prediction'].values > opt_thresh
    visits_progressed = dl_preds['Progressor_visit'].map(lambda x: mapping_visits[x], 1).values
    progressed = dl_preds['Progressor'] > 0

    plt.figure()
    kmf = KaplanMeierFitter()
    kmf.fit(visits_progressed[groups == 0], event_observed=progressed[groups == 0], label='non-progressed (pred)')
    ax = kmf.plot()
    kmf.fit(visits_progressed[groups == 1], event_observed=progressed[groups == 1], label='progressed (pred)')
    ax = kmf.plot(ax=ax)
    plt.xlim(0, 84)
    plt.show()
