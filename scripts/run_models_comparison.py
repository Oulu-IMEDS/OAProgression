import os

if int(os.getenv('USE_AGG', 1)) == 1:
    import matplotlib

    matplotlib.use('Agg')

import argparse
import pandas as pd
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt

from oaprogression.evaluation import stats
from oaprogression.evaluation.tools import pkl2df, init_auc_pr_plot, compute_and_plot_curves

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

    print(colored('====> ', 'green') + 'Logistic regression baselines')
    fig, axs = init_auc_pr_plot(dl_preds.Progressor.values)
    for key in models:
        if 'logreg' not in key or key == 'logreg_kl':
            continue

        tmp_df = models[key]
        key = key.split('logreg')[1]
        key = ' '.join(key.split('_')).upper()
        compute_and_plot_curves(tmp_df, axs, key=key, legend=True, seed=args.seed)

    plt.savefig(os.path.join(args.results_dir, 'Logreg_baselines.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)

    fig, axs = init_auc_pr_plot(dl_preds.Progressor.values)
    for key in models:
        if key != 'logreg_kl':
            continue

        tmp_df = models[key]
        key = key.split('logreg')[1]
        key = ' '.join(key.split('_')).upper()
        compute_and_plot_curves(tmp_df, axs, key=key, legend=True, seed=args.seed)
    plt.savefig(os.path.join(args.results_dir, 'KL_grade_baseline.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print(colored('====> ', 'green') + 'LightGBM baselines')
    fig, axs = init_auc_pr_plot(dl_preds.Progressor.values)
    for key in models:
        if 'lgbm' not in key or 'lgbm_stacking' in key:
            continue

        tmp_df = models[key]
        key = key.split('lgbm')[1]
        key = ' '.join(key.split('_')).upper()
        compute_and_plot_curves(tmp_df, axs, key=key, legend=True, seed=args.seed)

    plt.savefig(os.path.join(args.results_dir, 'Lgbm_baselines.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print(colored('====> ', 'green') + 'Plain CNN vs strongest baseline')
    fig, axs = init_auc_pr_plot(dl_preds.Progressor.values)
    for key, color, legend in zip(['lgbm_age_sex_bmi_kl_surg_inj_womac', 'dl'], ['blue', 'red'], ['GBM ref.', 'CNN']):
        tmp_df = models[key]
        compute_and_plot_curves(tmp_df, axs, key=legend, legend=True, color=color, seed=args.seed)

    plt.savefig(os.path.join(args.results_dir, 'DL_vs_strongest_baselines.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)

    tmp = pd.merge(models['lgbm_age_sex_bmi_kl_surg_inj_womac'], models['dl'].drop('Progressor', 1),
                   suffixes=['_lgb', '_dl'], on=('ID', 'Side'))
    logp = stats.delong_roc_test(tmp.Progressor, tmp.Prediction_dl, tmp.Prediction_lgb)
    print('P-value (DeLong AUC test):', 10 ** logp)

    print(colored('====> ', 'green') + 'Plain CNN vs logreg baseline')
    fig, axs = init_auc_pr_plot(dl_preds.Progressor.values)
    for key, color, legend in zip(['logreg_age_sex_bmi_kl_surg_inj_womac', 'dl'], ['blue', 'red'], ['LR ref.', 'CNN']):
        tmp_df = models[key]
        compute_and_plot_curves(tmp_df, axs, key=legend, legend=True, color=color, seed=args.seed)
    plt.savefig(os.path.join(args.results_dir, 'DL_vs_simple_baselines.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)

    tmp = pd.merge(models['logreg_age_sex_bmi_kl_surg_inj_womac'], models['dl'].drop('Progressor', 1),
                   suffixes=['_logreg', '_dl'], on=('ID', 'Side'))
    logp = stats.delong_roc_test(tmp.Progressor, tmp.Prediction_dl, tmp.Prediction_logreg)
    print('P-value (DeLong AUC test):', 10 ** logp)

    print(colored('====> ', 'green') + 'Best combination model vs strongest baseline (KL used)')
    fig, axs = init_auc_pr_plot(dl_preds.Progressor.values)
    for key, color, legend in zip(['lgbm_age_sex_bmi_kl_surg_inj_womac', 'lgbm_stacking_kl'], ['blue', 'red'],
                                  ['GBM ref.', 'Stacking w. KL']):
        tmp_df = models[key]
        compute_and_plot_curves(tmp_df, axs, key=legend, legend=True, color=color, seed=args.seed)
    plt.savefig(os.path.join(args.results_dir, 'Combined_vs_strongest_baseline.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print(colored('====> ', 'green') + 'Best combination model vs simplest baseline (KL used)')
    fig, axs = init_auc_pr_plot(dl_preds.Progressor.values)
    for key, color, legend in zip(['logreg_age_sex_bmi_kl_surg_inj_womac', 'lgbm_stacking_kl'], ['blue', 'red'],
                                  ['LR ref.', 'Stacking w. KL']):
        tmp_df = models[key]
        compute_and_plot_curves(tmp_df, axs, key=legend, legend=True, color=color, seed=args.seed)
    plt.savefig(os.path.join(args.results_dir, 'Combined_vs_simplest_baseline.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)

    tmp = pd.merge(models['lgbm_age_sex_bmi_kl_surg_inj_womac'], models['lgbm_stacking_kl'].drop('Progressor', 1),
                   suffixes=['_lgb', '_stacking'], on=('ID', 'Side'))
    logp = stats.delong_roc_test(tmp.Progressor, tmp.Prediction_lgb, tmp.Prediction_stacking)
    print('P-value (DeLong AUC test):', 10 ** logp)

    print(colored('====> ', 'green') + 'Combination model w/o KL vs strongest baseline')
    fig, axs = init_auc_pr_plot(dl_preds.Progressor.values)
    for key, color, legend in zip(['lgbm_age_sex_bmi_kl_surg_inj_womac', 'lgbm_stacking_no_kl'], ['blue', 'red'],
                                  ['GBM ref.', 'Stacking w/o KL']):
        tmp_df = models[key]
        compute_and_plot_curves(tmp_df, axs, key=legend, legend=True, color=color, seed=args.seed)

    plt.savefig(os.path.join(args.results_dir, 'Combined_automatic_vs_strongest_baseline.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)
    tmp = pd.merge(models['lgbm_age_sex_bmi_kl_surg_inj_womac'], models['lgbm_stacking_no_kl'].drop('Progressor', 1),
                   suffixes=['_stacking', '_lgb'], on=('ID', 'Side'))
    logp = stats.delong_roc_test(tmp.Progressor, tmp.Prediction_stacking, tmp.Prediction_lgb)
    print('P-value (DeLong AUC test):', 10 ** logp)

    print(colored('====> ', 'green') + 'Combined w/o KL vs simple baseline (KL used)')
    fig, axs = init_auc_pr_plot(dl_preds.Progressor.values)
    for key, color, legend in zip(['logreg_age_sex_bmi_kl_surg_inj_womac', 'lgbm_stacking_no_kl'], ['blue', 'red'],
                                  ['LR ref.', 'Stacking w/o KL']):
        tmp_df = models[key]
        compute_and_plot_curves(tmp_df, axs, key=legend, legend=True, color=color, seed=args.seed)

    plt.savefig(os.path.join(args.results_dir, 'Combined_automatic_vs_simple_baselines.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)
    tmp = pd.merge(models['logreg_age_sex_bmi_kl_surg_inj_womac'], models['lgbm_stacking_no_kl'].drop('Progressor', 1),
                   suffixes=['_dl', '_lgb'], on=('ID', 'Side'))
    logp = stats.delong_roc_test(tmp.Progressor, tmp.Prediction_dl, tmp.Prediction_lgb)
    print('P-value (DeLong AUC test):', 10 ** logp)
