import os

if int(os.getenv('USE_AGG', 1)) == 1:
    import matplotlib

    matplotlib.use('Agg')

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from oaprogression.evaluation import stats
from oaprogression.evaluation.tools import pkl2df
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve


def add_roc_curve(axs, tmp_df, key, color, legend=True, seed=12345, n_bootstrap=2000):
    auc, ci_l, ci_h, fpr, tpr = stats.calc_curve_bootstrap(roc_curve, roc_auc_score,
                                                           tmp_df.Progressor.values.astype(int),
                                                           tmp_df.Prediction.values.astype(float),
                                                           n_bootstrap=n_bootstrap,
                                                           seed=seed, stratified=True, alpha=95)
    if key is None:
        key = ''
    if color is None:
        axs.plot(fpr, tpr, label=key + f' ({np.round(auc, 2)} [{np.round(ci_l, 2)}, {np.round(ci_h, 2)}])')
    else:
        axs.plot(fpr, tpr, label=key + f' ({np.round(auc, 2)} [{np.round(ci_l, 2)}, {np.round(ci_h, 2)}])', color=color)
    if legend:
        axs.legend()


def add_pr_curve(axs, tmp_df, key, color, legend=True, seed=12345, n_bootstrap=2000):
    ap, ci_l, ci_h, precision, recall = stats.calc_curve_bootstrap(precision_recall_curve, average_precision_score,
                                                                   tmp_df.Progressor.values.astype(int),
                                                                   tmp_df.Prediction.values.astype(float),
                                                                   n_bootstrap=n_bootstrap,
                                                                   seed=seed, stratified=True, alpha=95)

    if color is None:
        axs.plot(recall, precision, label=key + f' ({np.round(ap, 2)} [{np.round(ci_l, 2)}, {np.round(ci_h, 2)}])')
    else:
        axs.plot(recall, precision, label=key + f' ({np.round(ap, 2)} [{np.round(ci_l, 2)}, {np.round(ci_h, 2)}])',
                 color=color)
    if legend:
        axs.legend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='')
    parser.add_argument('--metadata_root', default='')
    parser.add_argument('--seed', type=int, default=12345)
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
    dl_preds = dl_preds[['ID', 'Side', 'Progressor', 'Prediction']]

    bl_logreg = pkl2df(os.path.join(os.path.join(args.results_dir, 'results_baselines_logreg.pkl')))
    bl_lgbm = pkl2df(os.path.join(args.results_dir, 'results_baselines_lgbm.pkl'))
    lgbm_stacking = pkl2df(os.path.join(args.results_dir, 'results_lgbm_stacking.pkl'))

    models = dict()

    # logreg-based baselines
    models['logreg_age_sex_bmi'] = bl_logreg['preds_MOST_BL_all_AGE_SEX_BMI']
    models['logreg_kl'] = bl_logreg['preds_MOST_BL_all_KL']
    models['logreg_age_sex_bmi_kl'] = bl_logreg['preds_MOST_BL_all_AGE_SEX_BMI_KL']
    models['logreg_age_sex_bmi_surg_inj_womac'] = bl_lgbm['preds_MOST_BL_all_AGE_SEX_BMI_SURG_INJ_WOMAC']
    models['logreg_age_sex_bmi_kl_surg_inj_womac'] = bl_logreg['preds_MOST_BL_all_AGE_SEX_BMI_KL_SURG_INJ_WOMAC']

    # lgbm-based baselines
    models['lgbm_age_sex_bmi'] = bl_lgbm['preds_MOST_BL_all_AGE_SEX_BMI']
    models['lgbm_age_sex_bmi_kl'] = bl_lgbm['preds_MOST_BL_all_AGE_SEX_BMI_KL']
    models['lgbm_age_sex_bmi_surg_inj_womac'] = bl_lgbm['preds_MOST_BL_all_AGE_SEX_BMI_SURG_INJ_WOMAC']
    models['lgbm_age_sex_bmi_kl_surg_inj_womac'] = bl_lgbm['preds_MOST_BL_all_AGE_SEX_BMI_KL_SURG_INJ_WOMAC']

    # Just an image as an input: DL predictions as an output
    models['dl'] = dl_preds

    # Stacking (combination model)
    models['lgbm_stacking_no_kl'] = lgbm_stacking['preds_MOST_BL_all_AGE_SEX_BMI_SURG_INJ_WOMAC_'
                                                  'kl_pred_0_kl_pred_1_kl_pred_2_kl_pred_3_'
                                                  'prog_pred_0_prog_pred_1_prog_pred_2']

    models['lgbm_stacking_kl'] = lgbm_stacking['preds_MOST_BL_all_AGE_SEX_BMI_KL_SURG_INJ_WOMAC_'
                                               'kl_pred_0_kl_pred_1_kl_pred_2_kl_pred_3_'
                                               'prog_pred_0_prog_pred_1_prog_pred_2']

    for method in ['logreg', 'lgbm']:
        matplotlib.rcParams.update({'font.size': 14})
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        axs.plot([0, 1], [0, 1], '--', color='black')
        axs.set_xlim([0, 1])
        axs.set_ylim([0, 1])
        axs.grid()
        axs.set_xlabel('False positive rate')
        axs.set_ylabel('True positive rate')

        add_roc_curve(axs, models[f'{method}_age_sex_bmi_kl_surg_inj_womac'],
                      'Age, SEX, BMI, KL, Surg, Inj, WOMAC', 'r', seed=args.seed)

        add_roc_curve(axs, models[f'{method}_age_sex_bmi_kl'],
                      'Age, SEX, BMI, KL', 'g', seed=args.seed)

        add_roc_curve(axs, models[f'{method}_age_sex_bmi_surg_inj_womac'],
                      'Age, SEX, BMI, Surg, Inj, WOMAC', 'b', seed=args.seed)

        add_roc_curve(axs, models[f'{method}_age_sex_bmi'],
                      'Age, SEX, BMI', 'k', seed=args.seed)

        plt.savefig(os.path.join(args.results_dir, f'roc_curves_{method}.pdf'), bbox_inches='tight')
        plt.show()
        plt.close(fig)

        matplotlib.rcParams.update({'font.size': 14})
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        y = models[f'{method}_age_sex_bmi'].Progressor.values
        axs.axhline(y=y.sum() / y.shape[0], linestyle='--', color='black')
        axs.set_xlim([0, 1])
        axs.set_ylim([0, 1])
        axs.grid()
        axs.set_xlabel('Recall')
        axs.set_ylabel('Precision')

        add_pr_curve(axs, models[f'{method}_age_sex_bmi_kl_surg_inj_womac'],
                     'Age, SEX, BMI, KL, Surg, Inj, WOMAC', 'r', seed=args.seed)

        add_pr_curve(axs, models[f'{method}_age_sex_bmi_kl'],
                     'Age, SEX, BMI, KL', 'g', seed=args.seed)

        add_pr_curve(axs, models[f'{method}_age_sex_bmi_surg_inj_womac'],
                     'Age, SEX, BMI, Surg, Inj, WOMAC', 'b', seed=args.seed)

        add_pr_curve(axs, models[f'{method}_age_sex_bmi'],
                     'Age, SEX, BMI', 'k', seed=args.seed)

        plt.savefig(os.path.join(args.results_dir, f'pr_curves_{method}.pdf'), bbox_inches='tight')
        plt.show()
        plt.close(fig)

    matplotlib.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    axs.plot([0, 1], [0, 1], '--', color='black')
    axs.set_xlim([0, 1])
    axs.set_ylim([0, 1])
    axs.grid()
    axs.set_xlabel('False positive rate')
    axs.set_ylabel('True positive rate')

    add_roc_curve(axs, models['dl'],
                  'CNN', 'r', seed=args.seed)

    add_roc_curve(axs, models['lgbm_age_sex_bmi_kl_surg_inj_womac'],
                  'GBM ref', 'g', seed=args.seed)

    add_roc_curve(axs, models['logreg_age_sex_bmi_kl_surg_inj_womac'],
                  'LR ref.', 'b', seed=args.seed)

    plt.savefig(os.path.join(args.results_dir, f'roc_curves_cnn_vs_ref_methods.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)

    matplotlib.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    y = models[f'{method}_age_sex_bmi'].Progressor.values
    axs.axhline(y=y.sum() / y.shape[0], linestyle='--', color='black')
    axs.set_xlim([0, 1])
    axs.set_ylim([0, 1])
    axs.grid()
    axs.set_xlabel('Recall')
    axs.set_ylabel('Precision')

    add_pr_curve(axs, models['dl'],
                 'CNN', 'r', seed=args.seed)

    add_pr_curve(axs, models['lgbm_age_sex_bmi_kl_surg_inj_womac'],
                 'GBM ref', 'g', seed=args.seed)

    add_pr_curve(axs, models['logreg_age_sex_bmi_kl_surg_inj_womac'],
                 'LR ref.', 'b', seed=args.seed)

    plt.savefig(os.path.join(args.results_dir, f'pr_curves_cnn_vs_ref_methods.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)

    matplotlib.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    axs.plot([0, 1], [0, 1], '--', color='black')
    axs.set_xlim([0, 1])
    axs.set_ylim([0, 1])
    axs.grid()
    axs.set_xlabel('False positive rate')
    axs.set_ylabel('True positive rate')

    add_roc_curve(axs, models['lgbm_stacking_kl'],
                  'Stacking w. KL', 'c', seed=args.seed)

    add_roc_curve(axs, models['lgbm_stacking_no_kl'],
                  'Stacking w/o KL', 'r', seed=args.seed)

    add_roc_curve(axs, models['lgbm_age_sex_bmi_kl_surg_inj_womac'],
                  'GBM ref', 'g', seed=args.seed)

    add_roc_curve(axs, models['logreg_age_sex_bmi_kl_surg_inj_womac'],
                  'LR ref.', 'b', seed=args.seed)

    plt.savefig(os.path.join(args.results_dir, f'roc_curves_stacking_vs_ref_methods.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)

    matplotlib.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    y = models[f'{method}_age_sex_bmi'].Progressor.values
    axs.axhline(y=y.sum() / y.shape[0], linestyle='--', color='black')
    axs.set_xlim([0, 1])
    axs.set_ylim([0, 1])
    axs.grid()
    axs.set_xlabel('Recall')
    axs.set_ylabel('Precision')

    add_pr_curve(axs, models['lgbm_stacking_kl'],
                 'Stacking w. KL', 'c', seed=args.seed)

    add_pr_curve(axs, models['lgbm_stacking_no_kl'],
                 'Stacking w/o KL', 'r', seed=args.seed)

    add_pr_curve(axs, models['lgbm_age_sex_bmi_kl_surg_inj_womac'],
                 'GBM ref', 'g', seed=args.seed)

    add_pr_curve(axs, models['logreg_age_sex_bmi_kl_surg_inj_womac'],
                 'LR ref.', 'b', seed=args.seed)

    plt.savefig(os.path.join(args.results_dir, f'pr_curves_stacking_vs_ref_methods.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)
