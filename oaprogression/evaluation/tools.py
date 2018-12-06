import glob
import torch.nn.functional as F
import cv2
import os
import numpy as np

from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix, mean_squared_error, f1_score, average_precision_score

from functools import partial

import solt.transforms as slt
import solt.core as slc

import torch
from torch import nn
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from oaprogression.training import model
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


def init_fold(fold_id, session_snapshot, args, return_fc_kl=False):
    net = model.KneeNet(session_snapshot['args'][0].backbone, 0.5)
    snapshot_name = glob.glob(os.path.join(args.snapshots_root, args.snapshot, f'fold_{fold_id}*.pth'))[0]

    net.load_state_dict(torch.load(snapshot_name))

    features = nn.DataParallel(net.features[:-1])
    fc = nn.DataParallel(net.classifier_prog[-1])

    features.to('cuda')
    fc.to('cuda')

    features.eval()
    fc.eval()

    if return_fc_kl:
        fc_kl = nn.DataParallel(net.classifier_kl[-1])
        fc_kl.to('cuda')
        fc_kl.eval()

        return features, fc, fc_kl

    return features, fc


def init_loader(metadata, args):

    mean_vector, std_vector = session.init_mean_std(args.snapshots_root, None, None, None)

    norm_trf = tv_transforms.Normalize(torch.from_numpy(mean_vector).float(),
                                       torch.from_numpy(std_vector).float())

    os.makedirs(args.save_dir, exist_ok=True)

    tta_trf = tv_transforms.Compose([
        img_labels2solt,
        slc.Stream([
            slt.PadTransform(pad_to=(700, 700), padding='z'),
            slt.CropTransform(crop_size=(700, 700), crop_mode='c'),
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


def eval_batch(sample, features, fc_prog, fc_kl=None):
    with torch.no_grad():
        inputs = sample['img'].to("cuda")
        bs, ncrops, c, h, w = inputs.size()
        maps = features(inputs.view(-1, c, h, w))
        maps_avg = F.adaptive_avg_pool2d(maps, 1).view(maps.size(0), -1)
        out_prog = F.softmax(fc_prog(maps_avg), 1).view(bs, ncrops, -1).mean(1)

        if fc_kl is not None:
            out_kl = F.softmax(fc_kl(maps_avg), 1).view(bs, ncrops, -1).mean(1)
            return out_prog.to("cpu").numpy(), out_kl.to("cpu").numpy()

        return out_prog.to("cpu").numpy()


def calc_metrics(gt_progression, gt_kl, preds_progression, preds_kl):
    # Computing Validation metrics
    preds_progression_bin = preds_progression[:, 1:].sum(1)
    preds_kl_bin = preds_kl[:, 1:].sum(1)

    res = dict()
    res['cm_prog'] = confusion_matrix(gt_progression, preds_progression.argmax(1))
    res['cm_kl'] = confusion_matrix(gt_kl, preds_kl.argmax(1))
    res['auc_prog'] = roc_auc_score(gt_progression > 0, preds_progression_bin)
    res['kappa_prog'] = cohen_kappa_score(gt_progression, preds_progression.argmax(1), weights="quadratic")
    res['acc_prog'] = np.mean(res['cm_prog'].diagonal().astype(float) / res['cm_prog'].sum(axis=1))
    res['mse_prog'] = mean_squared_error(gt_progression, preds_progression.argmax(1))
    res['auc_oa'] = roc_auc_score(gt_kl > 1, preds_kl_bin)
    res['kappa_kl'] = cohen_kappa_score(gt_kl, preds_kl.argmax(1), weights="quadratic")
    res['acc_kl'] = np.mean(res['cm_kl'].diagonal().astype(float) / res['cm_kl'].sum(axis=1))
    res['mse_kl'] = mean_squared_error(gt_kl, preds_kl.argmax(1))
    res['f1_score_03_prog'] = f1_score(gt_progression > 0, preds_progression_bin > 0.3)
    res['f1_score_04_prog'] = f1_score(gt_progression > 0, preds_progression_bin > 0.5)
    res['f1_score_05_prog'] = f1_score(gt_progression > 0, preds_progression_bin > 0.6)
    res['ap_prog'] = average_precision_score(gt_progression > 0, preds_progression_bin)

    return res
