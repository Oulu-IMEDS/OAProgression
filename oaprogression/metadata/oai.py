import pandas as pd
from tqdm import tqdm
import os
from oaprogression.metadata.utils import read_sas7bdata_pd


def build_img_progression_meta(oai_src_dir):
    visits = ['00', '12', '24', '36', '72', '96']
    exam_codes = ['00', '01', '03', '05', '08', '10']
    # 0 - no progression within 84 months range
    # 1 - progression earlier than 60 months
    # 2 - progression later than 60 months
    mapping_prog = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2}
    KL_files = []
    for i, visit in enumerate(visits):
        print(f'==> Reading OAI {visit} visit')
        meta = read_sas7bdata_pd(os.path.join(oai_src_dir,
                                              'Semi-Quant Scoring_SAS',
                                              f'kxr_sq_bu{exam_codes[i]}.sas7bdat'))
        meta.fillna(-1, inplace=True)
        for c in meta.columns:
            meta[c.upper()] = meta[c]
        # Removing the TKR at the baseline
        if i == 0:
            meta = meta[meta[f'V{exam_codes[i]}XRKL'] != -1]
            meta = meta[meta[f'V{exam_codes[i]}XRKL'] < 5]
        meta = meta[meta[f'V{exam_codes[i]}XRKL'] <= 4]
        meta.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)

        meta['KL'] = meta[f'V{exam_codes[i]}XRKL']
        KL_files.append(meta[['ID', 'SIDE', 'KL']])

    bad_ids = [9076900, 9466244]  # couldn't annotate these images, seem to be broken
    id_set_last_fu = set(KL_files[-1].ID.values.astype(int).tolist())  # Subjects present at all FU

    for follow_up_id in range(1, len(KL_files)):
        KL_files[follow_up_id] = KL_files[follow_up_id].set_index(['ID', 'SIDE'])

    # looking for progressors
    progressors = []
    identified_prog = set()
    sides = [None, 'R', 'L']
    for _, knee in tqdm(KL_files[0].iterrows(), total=KL_files[0].shape[0], desc='Processing OAI:'):
        if int(knee.ID) in bad_ids:
            continue
        if int(knee.ID) in identified_prog:
            if identified_prog[int(knee.ID)] == sides[int(knee.SIDE)]:
                continue
        for follow_up_id in range(1, len(KL_files)):
            follow_up = KL_files[follow_up_id]
            ind = follow_up.index.isin([(knee.ID, knee.SIDE)])
            # If the patient is present during the follow-up
            if ind.any():
                old_kl = int(knee.KL)
                new_kl = int(follow_up[ind].KL.values[0])
                # Skipping the ones who were identified as progressors already
                if (int(knee.ID), sides[int(knee.SIDE)]) in identified_prog:
                        continue
                if 0 <= new_kl <= 4:
                    # If not TKR
                    if new_kl != 1 and new_kl > old_kl:
                        progressors.append([int(knee.ID), sides[int(knee.SIDE)], old_kl, new_kl-old_kl, follow_up_id])
                        identified_prog.update({(int(knee.ID), sides[int(knee.SIDE)]), })
                else:
                    # Adding only the TKR cases here
                    if new_kl == -1:
                        # We will denote it as 5
                        progressors.append([int(knee.ID), sides[int(knee.SIDE)], old_kl, 5-old_kl, follow_up_id])
                        identified_prog.update({(int(knee.ID), sides[int(knee.SIDE)]), })

    # Looking for non-progressors
    non_progressors = []
    for _, knee in tqdm(KL_files[0].iterrows(), total=KL_files[0].shape[0]):
        if (int(knee.ID), sides[int(knee.SIDE)]) in identified_prog:
            continue
        if int(knee.ID) not in id_set_last_fu:
            continue
        if int(knee.ID) in bad_ids:
            continue

        non_progressors.append([int(knee.ID), sides[int(knee.SIDE)], int(knee.KL), 0, 0])

    data = pd.DataFrame(data=progressors + non_progressors, columns=['ID', 'Side', 'KL', 'Prog_increase', 'Progressor'])
    data.Progressor = data.apply(lambda x: mapping_prog[x.Progressor], axis=1)
    return data


def build_clinical(oai_src_dir):
    data_enrollees = read_sas7bdata_pd(os.path.join(oai_src_dir, 'enrollees.sas7bdat'))
    data_clinical = read_sas7bdata_pd(os.path.join(oai_src_dir, 'allclinical00.sas7bdat'))

    clinical_data_oai = data_clinical.merge(data_enrollees, on='ID')

    # Age, Sex, BMI
    clinical_data_oai['SEX'] = 2 - clinical_data_oai['P02SEX']
    clinical_data_oai['AGE'] = clinical_data_oai['V00AGE']
    clinical_data_oai['BMI'] = clinical_data_oai['P01BMI']

    clinical_data_oai_left = clinical_data_oai.copy()
    clinical_data_oai_right = clinical_data_oai.copy()

    # Making side-wise metadata
    clinical_data_oai_left['Side'] = 'L'
    clinical_data_oai_right['Side'] = 'R'

    # Injury (ever had)
    clinical_data_oai_left['INJ'] = clinical_data_oai_left['P01INJL']
    clinical_data_oai_right['INJ'] = clinical_data_oai_right['P01INJR']

    # Surgery (ever had)
    clinical_data_oai_left['SURG'] = clinical_data_oai_left['P01KSURGL']
    clinical_data_oai_right['SURG'] = clinical_data_oai_right['P01KSURGR']

    # Total WOMAC score
    clinical_data_oai_left['WOMAC'] = clinical_data_oai_left['V00WOMTSL']
    clinical_data_oai_right['WOMAC'] = clinical_data_oai_right['V00WOMTSR']

    clinical_data_oai = pd.concat((clinical_data_oai_left, clinical_data_oai_right))
    clinical_data_oai.ID = clinical_data_oai.ID.values.astype(int)
    return clinical_data_oai[['ID', 'Side', 'AGE', 'SEX', 'BMI', 'INJ', 'SURG', 'WOMAC']]

