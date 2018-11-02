import numpy as np
import pandas as pd
import glob


import torch
import cv2

import multiprocessing
from tqdm import tqdm
import gc

from sklearn.preprocessing import OneHotEncoder

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


class RSDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_list, transforms):
        super(RSDataset, self).__init__()
        self.dataset = imgs_list
        self.transforms = transforms

    def __getitem__(self, ind):
        img_path = self.dataset[ind]
        img = cv2.imread(img_path, 0)
        fname = img_path.split('/')[-1]
        if 'L' in fname:
            img = cv2.flip(img, 1)
        img_trf = self.transforms(img)

        return {'I': img_trf, 'fname': img_path}

    def __len__(self):
        return len(self.dataset)


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


