import sys
import os
import cv2
import pickle
from termcolor import colored
from sklearn.model_selection import GroupKFold


from oaprogression.kvs import GlobalKVS
from oaprogression.metadata.oai import read_jsw_metadata_oai
from oaprogression.training import session, train_utils
from oaprogression.evaluation import tools

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

DEBUG = sys.gettrace() is not None

sides = [None, 'R', 'L']
JSW_features = ['V00JSW150', 'V00JSW175', 'V00JSW200', 'V00JSW225', 'V00JSW250', 'V00JSW275', 'V00JSW300',
                'V00LJSW700', 'V00LJSW725', 'V00LJSW750', 'V00LJSW775', 'V00LJSW800', 'V00LJSW825', 'V00LJSW850',
                'V00LJSW875', 'V00LJSW900']

if __name__ == "__main__":
    kvs = GlobalKVS()
    session.init_session()
    sites, metadata = read_jsw_metadata_oai(kvs['args'].metadata_root, kvs['args'].oai_data_root)

    base_snapshot_name = kvs['snapshot_name']
    for test_site in sites:
        # Creating a sub-snapshot for every site in OAI
        os.makedirs(os.path.join(kvs['args'].snapshots, base_snapshot_name, f'site_{test_site}'), exist_ok=True)
        kvs.update('snapshot_name', os.path.join(base_snapshot_name, f'site_{test_site}'))
        # Splitting the data to exclude the current site from training and keeping it only
        top_subj_train = metadata[metadata.V00SITE != test_site]
        top_subj_test = metadata[metadata.V00SITE == test_site]

        kvs.update('metadata', top_subj_train)
        kvs.update('metadata_test', top_subj_test)

        gkf = GroupKFold(n_splits=kvs['args'].n_folds)
        cv_split = []
        for x in gkf.split(top_subj_train, y=top_subj_train.Progressor, groups=top_subj_train.ID):
            cv_split.append(x)

        kvs.update('cv_split_all_folds', cv_split)
        session.init_data_processing()
        writers = session.init_folds()
        train_utils.train_folds(writers)

        print(colored('====> ', 'green') + f'Testing site {test_site}....')
        with open(os.path.join(kvs['args'].snapshots, base_snapshot_name, f'site_{test_site}', 'session.pkl'), 'rb') as f:
            session_snapshot = pickle.load(f)

        loader = tools.init_loader(top_subj_test, kvs['args'], kvs['args'].snapshots)
        # We need to update the save dir every time we change the site for evaluation
        save_dir_path = os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'inference')
        os.makedirs(save_dir_path, exist_ok=True)
        tools.run_test_inference(loader, session_snapshot,
                                 kvs['args'].snapshots,
                                 kvs['snapshot_name'], save_dir_path)


