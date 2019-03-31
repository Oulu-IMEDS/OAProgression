import os

if int(os.getenv('USE_AGG', 1)) == 1:
    import matplotlib

    matplotlib.use('Agg')

import argparse
import pandas as pd
import numpy as np

from oaprogression.evaluation import tools
from oaprogression.evaluation.tools import pkl2df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='')
    parser.add_argument('--metadata_root', default='')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    progression_meta = pd.read_csv(os.path.join(args.metadata_root, 'MOST_progression.csv'))

    data = np.load(os.path.join(args.results_dir, 'results.npz'))

    preds_prog = data['preds_prog']
    preds_kl = data['preds_kl']
    ids = data['ids']
    dl_preds = pd.DataFrame(data={'ID': list(map(lambda x: x.split('_')[0], ids)),
                                  'Side': list(map(lambda x: x.split('_')[1], ids)),
                                  'Prediction': preds_prog[:, 1:].sum(1)})

    dl_preds = pd.merge(dl_preds, progression_meta, on=('ID', 'Side'))
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

    for model_name in models:
        tmp = pd.merge(models[model_name], progression_meta[['ID', 'Side', 'KL']], on=('ID', 'Side'))
        tmp = tmp[(tmp.KL == 0) | (tmp.KL == 1)]
        tools.compute_curves_and_metrics(model_name, tmp, seed=args.seed)
