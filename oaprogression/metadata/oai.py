import pandas as pd
from tqdm import tqdm
import os
from oaprogression.metadata.utils import read_sas7bdata_pd


def build_img_progression_meta(oai_src_dir):
    visits = ['00', '12', '24', '36', '72', '96']
    exam_codes = ['00', '01', '03', '05', '08', '10']
    # 0 - no progression within 96 months range
    # 1 - progression earlier than 60 months
    # 2 - progression later than 60 and earlier than 72 months
    # 3 - progression later than 72 and earlier than 96 months
    mapping_prog = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3}
    KL_files = []
    for i, visit in enumerate(visits):
        print(f'==> Reading OAI {visit} visit')
        meta = read_sas7bdata_pd(os.path.join(oai_src_dir,
                                              'Semi-Quant Scoring_SAS',
                                              f'kxr_sq_bu{exam_codes[i]}.sas7bdat'))
        meta.fillna(-1, inplace=True)
        for c in meta.columns:
            meta[c.upper()] = meta[c]
        if i == 0:
            meta = meta[meta[f'V{exam_codes[i]}XRKL'] != -1]
            meta = meta[meta[f'V{exam_codes[i]}XRKL'] < 5]
        meta = meta[meta[f'V{exam_codes[i]}XRKL'] <= 4]
        meta.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)

        meta = meta.assign(KL=meta[f'V{exam_codes[i]}XRKL'])
        KL_files.append(meta[['ID', 'SIDE', 'KL']])

    bad_ids = [9076900, 9466244]  # couldn't annotate these images, seem to be broken
    id_set_last_fu = set(KL_files[-1].ID.values.astype(int).tolist())  # Subjects present at all FU

    # looking for progressors
    progressors = []
    identified_prog = {}
    sides = [None, 'R', 'L']
    for _, knee in tqdm(KL_files[0].iterrows(), total=KL_files[0].shape[0], desc='Processing OAI:'):
        if int(knee.ID) in bad_ids:
            continue
        if int(knee.ID) in identified_prog:
            if identified_prog[int(knee.ID)] == sides[int(knee.SIDE)]:
                continue
        for follow_up_id in range(1, len(KL_files)):
            follow_up = KL_files[follow_up_id]
            follow_up = follow_up.set_index(['ID', 'SIDE'])
            ind = follow_up.index.isin([(knee.ID, knee.SIDE)])
            # If the patient is present during the follow-up
            if ind.any():
                old_kl = int(knee.KL)
                new_kl = int(follow_up[ind].KL.values[0])
                if 0 <= new_kl <= 4:
                    # If not TKR
                    if new_kl != 1 and new_kl > old_kl:
                        progressors.append([int(knee.ID), sides[int(knee.SIDE)], old_kl, follow_up_id])
                        identified_prog.update({int(knee.ID): sides[int(knee.SIDE)]})
                else:
                    # Adding only the TKR cases here
                    if new_kl == -1:
                        # We will denote it as 5
                        progressors.append([int(knee.ID), sides[int(knee.SIDE)], old_kl, follow_up_id])
                        identified_prog.update({int(knee.ID): sides[int(knee.SIDE)]})

    # Looking for non-progressors
    non_progressors = []
    for _, knee in tqdm(KL_files[0].iterrows(), total=KL_files[0].shape[0]):
        if int(knee.ID) in identified_prog:
            if identified_prog[int(knee.ID)] == sides[int(knee.SIDE)]:
                continue
        if int(knee.ID) not in id_set_last_fu:
            continue

        if int(knee.ID) in bad_ids:
            continue

        non_progressors.append([int(knee.ID), sides[int(knee.SIDE)], int(knee.KL), 0])

    data = pd.DataFrame(data=progressors + non_progressors, columns=['ID', 'Side', 'KL', 'Progressor'])
    data.Progressor = data.apply(lambda x: mapping_prog[x.Progressor], axis=1)
    return data


def build_clinical(oai_src_dir):
    data_enrollees = read_sas7bdata_pd(os.path.join(oai_src_dir, 'enrollees.sas7bdat'))
    data_clinical = read_sas7bdata_pd(os.path.join(oai_src_dir, 'allclinical00.sas7bdat'))
    data_clinical = data_clinical.merge(data_enrollees, on='ID')
    return data_clinical[['ID', 'V00AGE', 'P02SEX', 'P01BMI']]

