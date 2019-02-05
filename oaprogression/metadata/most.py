import glob
import os

import cv2
import pandas as pd
from tqdm import tqdm

from oaprogression.metadata.utils import read_sas7bdata_pd


def build_img_progression_meta(most_src_dir, img_dir):
    # 0 - no progression observed (up to 84 months)
    # 1 - progression earlier than 60 months
    # 2 - progression later than 60 and earlier than 84 months
    mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}

    files = glob.glob(os.path.join(most_src_dir, '*enroll.sas7bdat'))
    files_dict = {file.split('/')[-1].lower(): file for file in files}
    data = read_sas7bdata_pd(os.path.join(most_src_dir, 'mostv01235xray.sas7bdat')).fillna(-1)
    most_outcomes = read_sas7bdata_pd(os.path.join(most_src_dir, 'mostoutcomes.sas7bdat')).fillna(-1)
    ids_alive = most_outcomes[most_outcomes.V99EDINDEX == -1][['MOSTID']]
    # Excluding the people who died
    data = pd.merge(ids_alive, data)
    # Helpful for further i
    tkr_l = most_outcomes[most_outcomes['V99ELKRINDEX'] > 0][['MOSTID', 'V99ELKRINDEX']]
    tkr_r = most_outcomes[most_outcomes['V99ERKRINDEX'] > 0][['MOSTID', 'V99ERKRINDEX']]

    data.set_index('MOSTID', inplace=True)
    tkr_l.set_index('MOSTID', inplace=True)
    tkr_r.set_index('MOSTID', inplace=True)

    enrolled = {}
    for visit in [0, 1, 2, 3, 5]:
        print(f'==> Reading MOST {visit} visit')
        ds = read_sas7bdata_pd(files_dict[f'mostv{visit}enroll.sas7bdat'])
        ds = ds[ds['V{}PA'.format(visit)] == 1]  # Filtering out the cases when X-rays wern't taken
        ds = pd.merge(ids_alive, ds)
        id_set = set(ds.MOSTID.values.tolist())
        enrolled[visit] = id_set

    last_follow_up = enrolled[5]

    progressors = []
    non_progressors = []

    for ID in tqdm(enrolled[0], total=len(enrolled[0]), desc='Processing MOST:'):
        tmp_l = []
        tmp_r = []

        subj = data.loc[ID]
        KL_bl_l = subj['V{0}X{1}{2}'.format(0, 'L', 'KL')]
        KL_bl_r = subj['V{0}X{1}{2}'.format(0, 'R', 'KL')]
        for visit_id in [1, 2, 3, 5]:
            if ID in enrolled[visit_id]:
                KL_l = subj['V{0}X{1}{2}'.format(visit_id, 'L', 'KL')]
                KL_r = subj['V{0}X{1}{2}'.format(visit_id, 'R', 'KL')]

                tmp_l.append([ID, 'L', KL_bl_l, KL_l, visit_id])
                tmp_r.append([ID, 'R', KL_bl_r, KL_r, visit_id])

        # KL4 subjects are end-stage and do not progress. TKR can be made by other reasons
        if 0 <= KL_bl_l < 4 and cv2.imread(os.path.join(img_dir, f'{ID}_00_L.png')) is not None:
            if len(tmp_l) > 0:
                # We exclude missing values and also "grade 9"
                if sum(list(map(lambda x: x[3] == -1 or x[3] == 9, tmp_l))) == 0:
                    prog = None
                    # going through the follow-up grades
                    for point in tmp_l:
                        # if we notice a progression case, we stop looking and store the
                        # visit id where we stopped
                        if KL_bl_l < point[-2] < 9 and point[-2] != 1 and point[-2] != 1.9:
                            # This additional check is needed,
                            # because 8 can mean that the image could have been bad
                            if point[-2] == 8 and ID in tkr_l.index:
                                if tkr_l.loc[ID].V99ELKRINDEX == point[-1]:
                                    prog = point
                                    break
                            else:
                                prog = point
                                break
                    # Checking whether the case is a progressor
                    if prog is None:
                        # To ignore the patients who dropped during the study, we
                        # make sure that they were examined at the last follow-up
                        if ID in last_follow_up and subj['V{0}X{1}{2}'.format(5, 'R', 'KL')] < 5:
                            non_progressors.append([ID, 'L', KL_bl_l, 0, 0])
                    else:
                        progressors.append([ID, 'L', KL_bl_l, prog[-2] - KL_bl_l, prog[-1]])

        # Doing the same thing for the right knee
        if 0 <= KL_bl_r < 4 and cv2.imread(os.path.join(img_dir, f'{ID}_00_R.png')) is not None:
            if len(tmp_r) > 0:
                if sum(list(map(lambda x: x[3] == -1 or x[3] == 9, tmp_r))) == 0:
                    prog = None
                    for point in tmp_r:
                        # if we notice a progression case, we stop looking and store the
                        # visit id where we stopped
                        if KL_bl_r < point[-2] < 9 and point[-2] != 1 and point[-2] != 1.9:
                            # This additional check is needed,
                            # because 8 can mean that the image could have been bad
                            if point[-2] == 8 and ID in tkr_r.index:
                                if tkr_r.loc[ID].V99ERKRINDEX == point[-1]:
                                    prog = point
                                    break
                            else:
                                prog = point
                                break
                    if prog is None:
                        if ID in last_follow_up and subj['V{0}X{1}{2}'.format(5, 'R', 'KL')] < 5:
                            non_progressors.append([ID, 'R', KL_bl_r, 0, 0])
                    else:
                        progressors.append([ID, 'R', KL_bl_r, prog[-2] - KL_bl_r, prog[-1]])

    progr_data = pd.DataFrame(progressors + non_progressors,
                              columns=['ID', 'Side', 'KL', 'Prog_increase', 'Progressor'])
    progr_data['Progressor_visit'] = progr_data.Progressor.copy()
    progr_data.Progressor = progr_data.apply(lambda x: mapping[x.Progressor], 1)
    return progr_data


def build_clinical(most_src_dir):
    files = glob.glob(os.path.join(most_src_dir, '*enroll.sas7bdat'))
    files_dict = {file.split('/')[-1].lower(): file for file in files}
    clinical_data_most = read_sas7bdata_pd(files_dict['mostv0enroll.sas7bdat'])
    clinical_data_most['ID'] = clinical_data_most.MOSTID
    clinical_data_most['BMI'] = clinical_data_most['V0BMI']

    clinical_data_most_left = clinical_data_most.copy()
    clinical_data_most_right = clinical_data_most.copy()

    # Making side-wise metadata
    clinical_data_most_left['Side'] = 'L'
    clinical_data_most_right['Side'] = 'R'

    # Injury (ever had)
    clinical_data_most_left['INJ'] = clinical_data_most_left['V0LAL']
    clinical_data_most_right['INJ'] = clinical_data_most_right['V0LAR']

    # Surgery (ever had)
    clinical_data_most_left['SURG'] = clinical_data_most_left['V0SURGL']
    clinical_data_most_right['SURG'] = clinical_data_most_right['V0SURGR']

    # Total WOMAC score
    clinical_data_most_left['WOMAC'] = clinical_data_most_left['V0WOTOTL']
    clinical_data_most_right['WOMAC'] = clinical_data_most_right['V0WOTOTR']

    clinical_data_most = pd.concat((clinical_data_most_left, clinical_data_most_right))

    return clinical_data_most[['ID', 'Side', 'AGE', 'SEX', 'BMI', 'INJ', 'SURG', 'WOMAC']]
