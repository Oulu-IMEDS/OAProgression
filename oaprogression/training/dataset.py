import torch
import torch.utils.data as data
import cv2
import os
import pandas as pd
from oaprogression.kvs import GlobalKVS
from sklearn.model_selection import GroupKFold
from termcolor import colored

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


class OAProgressionDataset(data.Dataset):
    def __init__(self, dataset, split, transforms):
        self.dataset = dataset
        self.split = split
        self.transforms = transforms

    def __getitem__(self, ind):
        if isinstance(ind, torch.Tensor):
            ind = ind.item()
        entry = self.split.iloc[ind]
        fname = os.path.join(self.dataset, '{}_00_{}.png'.format(entry.ID, entry.Side))
        img = cv2.imread(fname, 0)
        if entry.Side == 'L':
            img = cv2.flip(img, 1)
        img = img.reshape((img.shape[0], img.shape[1], 1))
        img, kl_grade, label = self.transforms((img, entry.KL, entry.Progressor))

        res = {'KL': kl_grade,
               'img': img,
               'label': label,
               'ID_SIDE': str(entry.ID) + '_' + entry.Side
               }

        return res

    def __len__(self):
        return self.split.shape[0]


def make_weights_for_balanced_classes(labels):
    """
    https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3

    """
    nclasses = max(labels) + 1
    count = [0] * nclasses

    for l in labels:
        count[l] += 1
    weight_per_class = [0.] * nclasses
    N = float(len(labels))

    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]
    return weight, weight_per_class


def init_metadata():
    # We get should rid of non-progressors from MOST because we can check
    # non-progressors only up to 84 months
    kvs = GlobalKVS()

    most_meta_full = pd.read_csv(os.path.join(kvs['args'].metadata_root, 'MOST_progression.csv'))
    oai_meta = pd.read_csv(os.path.join(kvs['args'].metadata_root, 'OAI_progression.csv'))

    most_meta = most_meta_full#[most_meta_full.Progressor > 0]
    kvs.update('metadata', pd.concat((oai_meta, most_meta), axis=0))

    gkf = GroupKFold(n_splits=5)
    cv_split = [x for x in gkf.split(kvs['metadata'],
                                     kvs['metadata']['Progressor'],
                                     kvs['metadata']['ID'].astype(str))]

    kvs.update('cv_split_all_folds', cv_split)
    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))

    print(colored("==> ", 'green') + f"Combined dataset has "
                                     f"{(kvs['metadata'].Progressor == 0).sum()} non-progressed knees")

    print(colored("==> ", 'green')+f"Combined dataset has "
                                   f"{(kvs['metadata'].Progressor > 0).sum()} progressed knees")