import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
from oaprogression.metadata.utils import read_sas7bdata_pd


def build_img_progression_meta(most_src_dir):
    # 0 - no progression observed (up to 84 months)
    # 1 - progression earlier than 60 months
    # 2 - progression later than 60 and earlier than 84 months
    mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}

    files = glob.glob(os.path.join(most_src_dir, '*enroll.sas7bdat'))
    files_dict = {file.split('/')[-1].lower(): file for file in files}
    most_names = np.loadtxt(os.path.join(most_src_dir, 'MOST_names.csv'), dtype=str)
    data = read_sas7bdata_pd(os.path.join(most_src_dir, 'mostv01235xray.sas7bdat')).fillna(-1)
    data.set_index('MOSTID', inplace=True)
    pa_10_bl_ids = set([fname[:5] for fname in most_names if ('V0' in fname and 'PA10' in fname)])

    enrolled = {}
    for visit in [0, 1, 2, 3, 5]:
        print(f'==> Reading MOST {visit} visit')
        ds = read_sas7bdata_pd(files_dict[f'mostv{visit}enroll.sas7bdat'])
        ds = ds[ds['V{}PA'.format(visit)] == 1]  # Filtering out the cases when X-rays wern't taken
        id_set = set(ds.MOSTID.values.tolist())
        enrolled[visit] = id_set

    last_follow_up = enrolled[5]

    progressors = []
    non_progressors = []

    for ID in tqdm(enrolled[0], total=len(enrolled[0]), desc='Processing MOST:'):
        if ID not in pa_10_bl_ids:
            continue
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

        if 0 <= KL_bl_l <= 4:
            if len(tmp_l) > 0:
                # We exclude missing values and also "grades" 9 and 8
                if sum(list(map(lambda x: x[3] == -1 or x[3] == 8 or x[3] == 9, tmp_l))) == 0:
                    prog = None
                    for point in tmp_r:
                        if point[-2] > KL_bl_l and point[-2] != 1 and point[-2] <= 4 and point[-2] != 1.9:
                            prog = point
                            break
                    if prog is None:
                        if ID in last_follow_up:
                            non_progressors.append([ID, 'L', KL_bl_l, 0])
                    else:
                        progressors.append([ID, 'L', KL_bl_l, prog[-1]])

        if 0 <= KL_bl_r <= 4:
            if len(tmp_r) > 0:
                if sum(list(map(lambda x: x[3] == -1 or x[3] == 8 or x[3] == 9, tmp_r))) == 0:
                    prog = None
                    for point in tmp_r:
                        if point[-2] > KL_bl_r and point[-2] != 1 and point[-2] <= 4 and point[-2] != 1.9:
                            prog = point
                            break
                    if prog is None:
                        if ID in last_follow_up:
                            non_progressors.append([ID, 'R', KL_bl_r, 0])
                    else:
                        progressors.append([ID, 'R', KL_bl_r, prog[-1]])

    progr_data = pd.DataFrame(progressors + non_progressors, columns=['ID', 'Side', 'KL', 'Progressor'])
    progr_data.Progressor = progr_data.apply(lambda x: mapping[x.Progressor], 1)
    return progr_data


def build_clinical(most_src_dir):
    files = glob.glob(os.path.join(most_src_dir, '*enroll.sas7bdat'))
    files_dict = {file.split('/')[-1].lower(): file for file in files}
    data_enroll = read_sas7bdata_pd(files_dict['mostv0enroll.sas7bdat'])
    data_enroll['ID'] = data_enroll.MOSTID
    return data_enroll[['ID', 'AGE', 'SEX', 'V0BMI']]
