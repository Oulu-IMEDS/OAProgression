import sys
import os
import cv2
import pandas as pd
import argparse
import numpy as np
from oaprogression.metadata.utils import read_sas7bdata_pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from oaprogression.training.lgbm_tools import optimize_lgbm_hyperopt, fit_lgb
import pickle
from oaprogression.evaluation import tools

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

DEBUG = sys.gettrace() is not None

sides = [None, 'R', 'L']
JSW_features = ['V00JSW150', 'V00JSW175', 'V00JSW200', 'V00JSW225', 'V00JSW250', 'V00JSW275', 'V00JSW300',
                'V00LJSW700', 'V00LJSW725', 'V00LJSW750', 'V00LJSW775', 'V00LJSW800', 'V00LJSW825', 'V00LJSW850',
                'V00LJSW875', 'V00LJSW900']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='')
    parser.add_argument('--metadata_root', default='')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--lgbm_hyperopt_trials', type=int, default=500)
    parser.add_argument('--save_dir', default='')
    args = parser.parse_args()

    oai_meta = pd.read_csv(os.path.join(args.metadata_root, 'OAI_progression.csv'))
    oai_participants = pd.read_csv(os.path.join(args.metadata_root, 'OAI_participants.csv'))
    oai_participants_raw = read_sas7bdata_pd(os.path.join(os.path.join(args.dataset_root,
                                                                       'X-Ray_Image_Assessments_SAS'),
                                                          'enrollees.sas7bdat'))

    sites = oai_participants_raw[['ID', 'V00SITE']]
    sites.ID = sites.ID.astype(int)
    metadata = pd.merge(oai_meta, oai_participants, on=('ID', 'Side'))
    metadata = pd.merge(metadata, sites)

    quant_readings = read_sas7bdata_pd(os.path.join(args.dataset_root, 'X-Ray_Image_Assessments_SAS',
                                                    'Quant JSW_SAS',
                                                    'kxr_qjsw_duryea00.sas7bdat'))

    quant_readings.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)
    quant_readings = quant_readings[(quant_readings['V00NOLJSWX'].astype(float) +
                                     quant_readings['V00NOMJSWX'].astype(float)) == 0]

    quant_readings = quant_readings[['ID', 'SIDE'] + JSW_features+ ['V00BMANG']]

    quant_readings['Side'] = quant_readings.SIDE.apply(lambda x: (sides[int(x)]), 1)
    quant_readings['ID'] = quant_readings.ID.astype(int)
    quant_readings.drop('SIDE', axis=1, inplace=True)
    metadata = pd.merge(quant_readings, metadata, on=('ID', 'Side'))
    sites = np.unique(metadata.V00SITE.values)

    results = {}
    for feature_set in [['AGE', 'SEX', 'BMI'], # Reproducing the test results
                        ['AGE', 'SEX', 'BMI', 'SURG', 'INJ', 'WOMAC'],
                        ['AGE', 'SEX', 'BMI', 'KL'],
                        ['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC'],
                        ['AGE', 'SEX', 'BMI', 'V00BMANG'], # Reproducing the test results w. beam angle
                        ['AGE', 'SEX', 'BMI', 'SURG', 'INJ', 'WOMAC', 'V00BMANG'],
                        ['AGE', 'SEX', 'BMI', 'KL', 'V00BMANG'],
                        ['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC', 'V00BMANG'],
                        ['AGE', 'SEX', 'BMI'] + JSW_features, # Adding JSW to the base model
                        ['AGE', 'SEX', 'BMI', 'KL'] + JSW_features,
                        ['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC'] + JSW_features,
                        ['AGE', 'SEX', 'BMI', 'SURG', 'INJ', 'WOMAC'] + JSW_features,
                        ['AGE', 'SEX', 'BMI', 'V00BMANG'] + JSW_features, # Let's try to add the beam angle as well
                        ['AGE', 'SEX', 'BMI', 'KL', 'V00BMANG'] + JSW_features,
                        ['AGE', 'SEX', 'BMI', 'KL', 'SURG', 'INJ', 'WOMAC', 'V00BMANG'] + JSW_features,
                        ['AGE', 'SEX', 'BMI', 'SURG', 'INJ', 'WOMAC', 'V00BMANG'] + JSW_features,
                        ]:

        features_suffix = '_'.join(feature_set)
        results[features_suffix] = {}
        for test_site in sites:
            top_subj_train = metadata[metadata.V00SITE != test_site]
            top_subj_test = metadata[metadata.V00SITE == test_site]

            gkf = GroupKFold(n_splits=5)
            train_folds = []
            for train_idx, val_idx in gkf.split(top_subj_train, y=top_subj_train.Progressor, groups=top_subj_train.ID):
                train_folds.append((top_subj_train.iloc[train_idx], top_subj_train.iloc[val_idx]))

            best_params, trials = optimize_lgbm_hyperopt(train_folds, feature_set,
                                                         average_precision_score,
                                                         args.seed, hyperopt_trials=args.lgbm_hyperopt_trials)

            ap_score, models_best, oof_preds = fit_lgb(best_params, train_folds,
                                                       feature_set, average_precision_score, True, True)

            print('CV score:', feature_set, ap_score)
            test_res = tools.eval_models(top_subj_test, feature_set, models_best, mean_std_best=None,
                                         impute=False, model_type='lgbm')

            y_test = top_subj_test.Progressor.values.copy() > 0
            ids = top_subj_test.ID.values
            sides = top_subj_test.Side.values

            results[features_suffix][test_site] = (ids, sides, y_test, test_res)

    with open(os.path.join(args.save_dir, 'results_ablation_jsw_lgbm.pkl'), 'wb') as f:
        pickle.dump(results, f)
