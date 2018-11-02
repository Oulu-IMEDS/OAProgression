import glob
import os
import argparse
import pandas as pd

from oaprogression.kvs import GlobalKVS
from oaprogression. evaluation import rstools
from oaprogression.training import session as session


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/data/DL_spring2/OA_progression_project/Data/RS_data/')
    parser.add_argument('--rs_cohort', default=3)
    parser.add_argument('--snapshots_dir', default='/data/DL_spring2/OA_progression_project/snapshots')
    args = parser.parse_args()

    mean_vect, std_vect = session.init_mean_std(args.snapshots_dir, None, None, None)

