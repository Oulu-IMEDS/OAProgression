import pandas as pd

import glob
import copy

import cv2
import os

from oaprogression.training import model

from functools import partial

import solt.transforms as slt
import solt.core as slc

import torch
from torch import nn
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from oaprogression.training import session as session
from oaprogression.training.dataset import OAProgressionDataset, unpack_solt_data, img_labels2solt, apply_by_index


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def five_crop(img, size):
    """Returns a stacked 5 crop
    """
    img = img.clone()
    h, w = img.size()[-2:]
    # get central crop
    c_cr = img[:, h//2-size//2:h//2+size//2, w//2-size//2:w//2+size//2]
    # upper-left crop
    ul_cr = img[:, 0:size, 0:size]
    # upper-right crop
    ur_cr = img[:, 0:size, w-size:w]
    # bottom-left crop
    bl_cr = img[:, h-size:h, 0:size]
    # bottom-right crop
    br_cr = img[:, h-size:h, w-size:w]
    return torch.stack((c_cr, ul_cr, ur_cr, bl_cr, br_cr))


def init_fold(fold_id, session_snapshot, args):
    net = model.KneeNet(session_snapshot['args'][0].backbone, 0.5)
    snapshot_name = glob.glob(os.path.join(args.snapshots_root, args.snapshot, f'fold_{fold_id}*.pth'))[0]

    net.load_state_dict(torch.load(snapshot_name))

    features = nn.DataParallel(net.features[:-1])
    fc = nn.DataParallel(net.classifier_prog[-1])

    features.to('cuda')
    fc.to('cuda')

    features.eval()
    fc.eval()

    return features, fc


def init_loader(metadata, args):

    mean_vector, std_vector = session.init_mean_std(args.snapshots_root, None, None, None)

    norm_trf = tv_transforms.Normalize(torch.from_numpy(mean_vector).float(),
                                       torch.from_numpy(std_vector).float())

    os.makedirs(args.save_dir, exist_ok=True)

    tta_trf = tv_transforms.Compose([
        img_labels2solt,
        slc.Stream([
            slt.PadTransform(pad_to=(700,700), padding='z'),
            slt.CropTransform(crop_size=(700,700), crop_mode='c'),
            slt.ResizeTransform(resize_to=(310, 310), interpolation='bicubic'),
            slt.ImageColorTransform(mode='gs2rgb'),
        ], interpolation='bicubic'),
        unpack_solt_data,
        partial(apply_by_index, transform=tv_transforms.ToTensor(), idx=0),
        partial(apply_by_index, transform=norm_trf, idx=0),
        partial(apply_by_index, transform=partial(five_crop, size=300), idx=0),
    ])

    dataset = OAProgressionDataset(dataset=args.dataset_root,
                                   split=metadata, transforms=tta_trf)

    loader = DataLoader(dataset,
                        batch_size=args.bs,
                        sampler=SequentialSampler(dataset),
                        num_workers=args.n_threads)

    return loader
