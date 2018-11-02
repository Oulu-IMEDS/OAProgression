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

from sklearn.preprocessing import OneHotEncoder

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


class RSDataset(data.Dataset):
    def __init__(self, imgs_list, transforms):
        super(RSDataset, self).__init__()
        self.dataset = imgs_list
        self.transforms = transforms

    def __getitem__(self, ind):
        entry = self.dataset[ind]
        img = cv2.imread(entry.fname, 0)
        fname = img_path.split('/')[-1]
        if 'L' in fname:
            img = cv2.flip(img, 1)
        img_trf = self.transforms(img)

        return {'I': img_trf,
                'fname': img_path,
                'ergo_id': entry.ergo_id,
                'side': entry.side,
                'progressor': float(entry.progressor)
                }

    def __len__(self):
        return self.dataset.shape[0]


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


def preprocess_rs_meta(ds, rs_cohort):
    ds = copy.deepcopy(ds)
    all_null = ds.kll1.isnull() | ds.kll2.isnull()
    ds = ds[~all_null]
    all_null = ds.klr1.isnull() | ds.klr2.isnull()
    ds = ds[~all_null]

    ds = ds[ds.rs_cohort == rs_cohort]

    ds['date_of_birth'] = pd.to_datetime(ds['date_of_birth'])
    ds['date1'] = pd.to_datetime(ds['date1'])
    ds['date2'] = pd.to_datetime(ds['date2'])
    ds['date3'] = pd.to_datetime(ds['date3'])
    ds['date4'] = pd.to_datetime(ds['date4'])

    L = ds[['ergoid', 'kll1',  'kll2']]
    L = L[L.apply(lambda x: x[1] <= 4, 1)]

    R = ds[['ergoid', 'klr1',  'klr2']]
    R = R[R.apply(lambda x: x[1] <= 4, 1)]

    R['progressor'] = R.apply(check_progression, 1)*1
    L['progressor'] = L.apply(check_progression, 1)*1

    L = pd.merge(L, ds.drop(['rs_cohort', 'kll1',  'kll2', 'kll3', 'kll4', 'klr1',
                             'klr2', 'klr3', 'klr4', 'date4', 'bmi4'], 1), how='left')
    R = pd.merge(R, ds.drop(['rs_cohort', 'kll1',  'kll2', 'kll3', 'kll4', 'klr1',
                             'klr2', 'klr3', 'klr4', 'date4', 'bmi4'], 1), how='left')

    L['side'] = 'L'
    R['side'] = 'R'

    L['kl1'] = L['kll1']
    R['kl1'] = R['klr1']

    L['kl2'] = L['kll2']
    R['kl2'] = R['klr2']

    rs_meta = pd.concat((L[['ergoid', 'side', 'kl1', 'kl2', 'progressor']],
                         R[['ergoid', 'side', 'kl1', 'kl2', 'progressor']]))
    return rs_meta
