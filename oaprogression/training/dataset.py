import torch.utils.data as data
import cv2
import os
import pandas as pd
from oaprogression.kvs import GlobalKVS

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


class OAProgressionDataset(data.Dataset):
    def __init__(self, dataset, split, transforms):
        self.dataset = dataset
        self.split = split
        self.transforms = transforms

    def __getitem__(self, ind):
        entry = self.split.iloc[ind]

        img = cv2.imread(self.dataset + '{}_00.png'.format(entry.ID))
        img, KL = self.transforms((img, entry.KL, entry.Progressor))

        res = {'KL': entry.KL,
               'img': img,
               'label': entry.Progressor,
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
    # We get rid of non-progressors from MOST because we can check
    # non-progressors only up to 84 months
    kvs = GlobalKVS()

    most_meta_full = pd.read_csv(os.path.join(kvs['args'].metadata_root, 'MOST_progression.csv'))
    oai_meta = pd.read_csv(os.path.join(kvs['args'].metadata_root, 'OAI_progression.csv'))

    most_meta = most_meta_full[most_meta_full.Progressor > 0]
    kvs.update('metadata', pd.concat((oai_meta, most_meta), axis=0))
    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))