import argparse
import os

import pandas as pd

from oaprogression.metadata import most
from oaprogression.metadata import oai
from oaprogression.metadata.utils import data_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--oai_meta',
                        default='/media/lext/FAST/OA_progression_project/Data/X-Ray_Image_Assessments_SAS')
    parser.add_argument('--most_meta', default='/media/lext/FAST/OA_progression_project/Data/most_meta')
    parser.add_argument('--imgs_dir', default='/media/lext/FAST/OA_progression_project/Data/MOST_OAI_00_0_2')
    parser.add_argument('--save_meta', default='/media/lext/FAST/OA_progression_project/workdir/Metadata/')
    args = parser.parse_args()

    os.makedirs(args.save_meta, exist_ok=True)
    if not os.path.isfile(os.path.join(args.save_meta, 'OAI_progression.csv')):
        oai_meta = oai.build_img_progression_meta(args.oai_meta)
        oai_meta.to_csv(os.path.join(args.save_meta, 'OAI_progression.csv'), index=None)
    else:
        oai_meta = pd.read_csv(os.path.join(args.save_meta, 'OAI_progression.csv'))
        print('OAI progression metadata exists!')

    if not os.path.isfile(os.path.join(args.save_meta, 'OAI_participants.csv')):
        oai_participants = oai.build_clinical(args.oai_meta)
        oai_participants.to_csv(os.path.join(args.save_meta, 'OAI_participants.csv'), index=None)
    else:
        oai_participants = pd.read_csv(os.path.join(args.save_meta, 'OAI_participants.csv'))
        print('OAI participants metadata exists!')

    if not os.path.isfile(os.path.join(args.save_meta, 'MOST_progression.csv')):
        most_meta = most.build_img_progression_meta(args.most_meta, args.imgs_dir)
        most_meta.to_csv(os.path.join(args.save_meta, 'MOST_progression.csv'), index=None)
    else:
        most_meta = pd.read_csv(os.path.join(args.save_meta, 'MOST_progression.csv'))
        print('MOST progression metadata exists!')

    if not os.path.isfile(os.path.join(args.save_meta, 'MOST_participants.csv')):
        most_participants = most.build_clinical(args.most_meta)
        most_participants.to_csv(os.path.join(args.save_meta, 'MOST_participants.csv'), index=None)
    else:
        most_participants = pd.read_csv(os.path.join(args.save_meta, 'MOST_participants.csv'))
        print('MOST participants metadata exists!')

    print(" ")
    print("# ======== OAI ======== ")
    data_stats(oai_meta, oai_participants)
    print(" ")
    print("# ======== MOST ======== ")
    data_stats(most_meta, most_participants)
