import argparse
import os
from oaprogression.metadata import oai
from oaprogression.metadata import most

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--oai_meta', default='')
    parser.add_argument('--most_meta', default='')
    parser.add_argument('--save_meta', default='Metadata/')
    args = parser.parse_args()

    os.makedirs(args.save_meta, exist_ok=True)
    if not os.path.isfile(os.path.join(args.save_meta, 'OAI_progression.csv')):
        oai_meta = oai.build_img_progression_meta(args.oai_meta)
        oai_meta.to_csv(os.path.join(args.save_meta, 'OAI_progression.csv'), index=None)
    else:
        print('OAI progression metadata exists!')

    if not os.path.isfile(os.path.join(args.save_meta, 'OAI_participants.csv')):
        oai_participants = oai.build_clinical(args.oai_meta)
        oai_participants.to_csv(os.path.join(args.save_meta, 'OAI_participants.csv'), index=None)
    else:
        print('OAI participants metadata exists!')

    if not os.path.isfile(os.path.join(args.save_meta, 'MOST_progression.csv')):
        most_meta = most.build_img_progression_meta(args.most_meta)
        most_meta.to_csv(os.path.join(args.save_meta, 'MOST_progression.csv'), index=None)
    else:
        print('MOST progression metadata exists!')

    if not os.path.isfile(os.path.join(args.save_meta, 'MOST_participants.csv')):
        most_meta = most.build_clinical(args.most_meta)
        most_meta.to_csv(os.path.join(args.save_meta, 'MOST_participants.csv'), index=None)
    else:
        print('MOST participants metadata exists!')
