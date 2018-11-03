import numpy as np
import pandas as pd
import glob

import copy

import torch
import cv2
import torch.utils.data as data
import multiprocessing
from tqdm import tqdm
import gc
import os
from sklearn.preprocessing import OneHotEncoder

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


class RSDataset(data.Dataset):
    def __init__(self, dataset_root, metadata, transforms):
        super(RSDataset, self).__init__()
        self.dataset_root = dataset_root
        self.metadata = metadata
        self.transforms = transforms

    def __getitem__(self, ind):
        entry = self.metadata.iloc[ind]
        img = cv2.imread(os.path.join(self.dataset_root, entry.fname), 0)
        if 'L' == entry.side:
            img = cv2.flip(img, 1)

        img_trf, kl, progressor = self.transforms((img, entry.kl, entry.progressor))

        return {'I': img_trf,
                'ergo_id': entry.ergo_id,
                'side': entry.side,
                'progressor': float(entry.progressor)
                }

    def __len__(self):
        return self.metadata.shape[0]


def five_crop(img, size):
    """
    Returns a stacked 5 crop
    """
    img = img.copy()
    h, w = img.size()[-2:]
    # get central crop
    c_cr = img[h//2-size//2:h//2+size//2, w//2-size//2:w//2+size//2]
    # upper-left crop
    ul_cr = img[0:size, 0:size]
    # upper-right crop
    ur_cr = img[0:size, w-size:w]
    # bottom-left crop
    bl_cr = img[h-size:h, 0:size]
    # bottom-right crop
    br_cr = img[h-size:h, w-size:w]
    return torch.stack((c_cr, ul_cr, ur_cr, bl_cr, br_cr))


def check_progression(x):
    _id, kl1, kl2 = x
    first = (kl2 > kl1) and kl2 != 1
    return first


def preprocess_rs_meta(ds, rs_meta_preselected, rs_cohort):
    ds = copy.deepcopy(ds)
    rs_meta_preselected = copy.deepcopy(rs_meta_preselected)

    ds.ergoid = ds.ergoid.astype(int)
    selected_ids = set(rs_meta_preselected.ergoid.values.astype(int).tolist())

    rs_meta_preselected = rs_meta_preselected.set_index('ergoid')

    ds['date_of_birth'] = pd.to_datetime(ds['date_of_birth'])
    ds['date1'] = pd.to_datetime(ds['date1'])
    ds['bmi'] = ds.bmi1
    ds['age'] = (ds['date1'] - ds['date_of_birth']) / pd.Timedelta('365 days')
    ds = ds[~ds.age.isnull()]
    ds = ds[~ds.bmi.isnull()]
    ds = ds[~ds.sex.isnull()]

    ds = ds[ds.rs_cohort == rs_cohort]

    # Cleaning the TKR at the baseline and mistakes
    L = ds[['ergoid', 'kll1',  'kll2']]
    L = L[~(L.kll1.isnull() | L.kll2.isnull())]

    L = L[L.apply(lambda x: (x[1] <= 4) and (x[1] <= x[2] if x[1] != 1 else True), 1)]

    # Cleaning the TKR at the baseline and mistakes
    R = ds[['ergoid', 'klr1',  'klr2']]
    R = R[~(R.klr1.isnull() | R.klr2.isnull())]

    R = R[R.apply(lambda x: (x[1] <= 4) and (x[1] <= x[2] if x[1] != 1 else True), 1)]

    R['progressor'] = R.apply(check_progression, 1)*1
    L['progressor'] = L.apply(check_progression, 1)*1

    L['side'] = 'L'
    R['side'] = 'R'

    L['kl1'] = L['kll1']
    R['kl1'] = R['klr1']

    L['kl2'] = L['kll2']
    R['kl2'] = R['klr2']

    rs_meta = pd.concat((L[['ergoid', 'side', 'kl1', 'kl2', 'progressor']],
                         R[['ergoid', 'side', 'kl1', 'kl2', 'progressor']]))

    take = []
    for _, entry in rs_meta.iterrows():
        if entry.ergoid in selected_ids:
            take.append(rs_meta_preselected.loc[entry.ergoid][entry.side])
        else:
            take.append(False)

    rs_meta = rs_meta[take]

    return rs_meta
