import os
# import matplotlib

# if int(os.getenv('USE_AGG', 1)) == 1:
#     import matplotlib
#
#     matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
import numpy as np
import shap
import pandas as pd


from sklearn.metrics import average_precision_score
from oaprogression.training.stacking import init_first_level_data_for_stacking
from oaprogression.training.lgbm_tools import fit_lgb

feature_remap_dict = {'AGE':'Age', 'SEX': 'Sex', 'BMI':'BMI', 'SURG':'Past Surgery',
                      'INJ': 'Past Injury',
                      'WOMAC': 'Total WOMAC score',
                      'KL': 'KL-grade',
                      'kl_pred_0': 'P(KL=0 | x)',
                      'kl_pred_1': 'P(KL=1 | x)',
                      'kl_pred_2': 'P(KL=2 | x)',
                      'kl_pred_3': 'P(KL=3 | x)',
                      'prog_pred_0': 'P(No-Progression | x)',
                      'prog_pred_1': 'P(Fast-Progression | x)',
                      'prog_pred_2': 'P(Slow-Progression | x)'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='')
    parser.add_argument('--metadata_root', default='')
    parser.add_argument('--results_dir', default='')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--snapshots_root', default='')
    parser.add_argument('--from_cache', default=False)
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--plot_gcams', type=bool, default=False)
    args = parser.parse_args()

    with open(os.path.join(args.snapshots_root, args.snapshot, 'session.pkl'), 'rb') as f:
        session_snapshot = pickle.load(f)

    train_folds, metadata_test = init_first_level_data_for_stacking(args, session_snapshot)

    with open(os.path.join(args.results_dir,  'results_lgbm_stacking.pkl'), 'rb') as f:
        data = pickle.load(f)

    for idx, feature_set in enumerate([['AGE', 'SEX', 'BMI', 'SURG', 'INJ', 'WOMAC', 'kl_pred_0', 'kl_pred_1', 'kl_pred_2',
                                        'kl_pred_3', 'prog_pred_0', 'prog_pred_1', 'prog_pred_2'],
                                       ['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC', 'kl_pred_0', 'kl_pred_1', 'kl_pred_2',
                                        'kl_pred_3', 'prog_pred_0', 'prog_pred_1', 'prog_pred_2']]):

        features_suffix = '_'.join(feature_set)

        _, best_params = data[f'lgbm_params_oof_preds_OAI_BL_all_{features_suffix}']

        ap_score, models_best, oof_preds = fit_lgb(best_params, train_folds,
                                                   feature_set, average_precision_score, True, True)

        x_test_initial = metadata_test[feature_set]
        explainers = []
        feature_importances = []

        for model in models_best:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_test_initial)
            explainers.append(shap_values[1])
            feature_importances.append(model.feature_importance())
        remapped_features = list(map(lambda x: feature_remap_dict[x], feature_set))
        # SHAP on the test set

        shap.summary_plot(np.mean(explainers, 0), x_test_initial, show=False, feature_names=remapped_features)
        plt.savefig(os.path.join(args.save_dir, f'FI_SHAP_{idx}.pdf'), bbox_inches='tight')
        plt.show()

        # Light GBM
        feature_importances = np.mean(feature_importances, 0)
        feature_importances /= feature_importances.sum()
        feature_importances *= 100
        feature_importances = np.round(feature_importances, 2)
        feature_imp = pd.DataFrame(sorted(zip(feature_importances, remapped_features)),
                                   columns=['Relative importance (%)', 'Feature'])

        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(8, 8))
        sns.barplot(x="Relative importance (%)",
                    y="Feature",
                    data=feature_imp.sort_values(by="Relative importance (%)", ascending=False))
        plt.xlim(0, 20)
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, f'FI_GBM_{idx}.pdf'), bbox_inches='tight')
        plt.show()
