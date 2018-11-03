import numpy as np
import pandas as pd

import torch.nn.functional as F
import glob
import copy

import cv2
import torch.utils.data as data
import os
from sklearn.preprocessing import OneHotEncoder
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
from oaprogression.training import dataset as dataset


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
        img = cv2.imread(os.path.join(self.dataset_root, f'{entry.ergoid}_{entry.side}.png'), 0)
        if 'L' == entry.side:
            img = cv2.flip(img, 1)

        img_trf, kl, progressor = self.transforms((img, entry.kl1, entry.progressor))

        return {'I': img_trf,
                'ergoid': entry.ergoid,
                'side': entry.side,
                'progressor': float(entry.progressor)
                }

    def __len__(self):
        return self.metadata.shape[0]


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


def eval_batch(sample, features, fc, ):
    # We don't need gradient to make an inference  for the features
    with torch.no_grad():
        inputs = sample['I'].to("cuda")
        bs, ncrops, c, h, w = inputs.size()
        maps = features(inputs.view(-1, c, h, w))

    fc.zero_grad()
    # Registering a hook to get the gradients
    grads = []
    maps_avg = F.adaptive_avg_pool2d(maps, 1).view(maps.size(0), -1)
    # First we should attach the variable back to the graph
    maps_avg.requires_grad = True
    # Now registering the backward hook
    maps_avg.register_hook(lambda x: grads.append(x))

    # Making the inference
    # Applying the TTA right away during the forward pass
    out_tmp = F.softmax(fc(maps_avg), 1).view(bs, ncrops, -1).mean(1)
    probs_not_summed = out_tmp.to("cpu").detach().numpy()
    # Summing the probabilities values for progression
    # This allows us to predict progressor / non-progressor
    out = torch.cat((out_tmp[:, 0].view(-1, 1), out_tmp[:, 1:].sum(1).view(-1, 1)), 1)
    # Saving the results to CPU
    probs = out.to("cpu").detach().numpy()

    # Using simple one hot encoder to create a fake gradient
    ohe = OneHotEncoder(sparse=False, n_values=out.size(1))
    # Creating the fake gradient (read the paper for details)
    index = np.argmax(probs, axis=1).reshape(-1, 1)
    fake_grad = torch.from_numpy(ohe.fit_transform(index)).float().to('cuda')
    # Backward pass after which we'll have the gradients
    out.backward(fake_grad)

    # Reshaping the activation maps sand getting the weights using the stored gradients
    # This way we would be able to consider GradCAM for each crop individually

    # Making the GradCAM
    # Going over the batch
    weight = grads[-1]
    with torch.no_grad():
        weighted_A = weight.unsqueeze(-1).unsqueeze(-1).expand(*maps.size()).mul(maps)
        gcam_batch = F.relu(weighted_A).view(bs, ncrops, -1, maps.size(-2), maps.size(-1)).sum(2)
        gcam_batch = gcam_batch.to('cpu').numpy()

    return gcam_batch, probs_not_summed


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


def init_loader(rs_meta, args):

    mean_vector, std_vector = session.init_mean_std(args.snapshots_root, None, None, None)

    norm_trf = tv_transforms.Normalize(torch.from_numpy(mean_vector).float(),
                                       torch.from_numpy(std_vector).float())

    os.makedirs(args.save_dir, exist_ok=True)

    tta_trf = tv_transforms.Compose([
        dataset.img_labels2solt,
        slc.Stream([
            slt.ResizeTransform((310, 310)),
            slt.ImageColorTransform(mode='gs2rgb'),
        ], interpolation='bicubic'),
        dataset.unpack_solt_data,
        partial(dataset.apply_by_index, transform=tv_transforms.ToTensor(), idx=0),
        partial(dataset.apply_by_index, transform=norm_trf, idx=0),
        partial(dataset.apply_by_index, transform=partial(five_crop, size=300), idx=0),
    ])

    rs_dataset = RSDataset(dataset_root=os.path.join(args.data_root, f'RS{args.rs_cohort}', 'localized'),
                           metadata=rs_meta,
                           transforms=tta_trf)

    loader = DataLoader(rs_dataset,
                        batch_size=args.bs,
                        sampler=SequentialSampler(rs_dataset),
                        num_workers=args.n_threads)

    return loader
