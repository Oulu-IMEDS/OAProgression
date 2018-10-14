import os
import time
import numpy as np
from tqdm import tqdm
from termcolor import colored
from functools import partial

import torch
from torch.utils.data import DataLoader

import solt.transforms as slt
from torchvision import transforms as tv_transforms

from oaprogression.training.args import parse_args
from oaprogression.training.dataset import OAProgressionDataset
from oaprogression.training.transforms import init_train_augs, apply_by_index, img_labels2solt, unpack_solt_data
from oaprogression.kvs import GlobalKVS, git_info


def init_session():
    kvs = GlobalKVS()

    # Getting the arguments
    args = parse_args()
    # Initializing the seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Creating the snapshot
    snapshot_name = time.strftime('%Y_%m_%d_%H_%M')
    os.makedirs(os.path.join(args.snapshots, snapshot_name), exist_ok=True)

    res = git_info()
    if res is not None:
        kvs.update('git branch name', res[0])
        kvs.update('git commit id', res[1])
    else:
        kvs.update('git branch name', None)
        kvs.update('git commit id', None)

    kvs.update('pytorch_version', torch.__version__)

    if torch.cuda.is_available():
        kvs.update('cuda', torch.version.cuda)
        kvs.update('gpus', torch.cuda.device_count())
    else:
        kvs.update('cuda', None)
        kvs.update('gpus', None)

    kvs.update('snapshot_name', snapshot_name)
    kvs.update('args', args)
    kvs.save_pkl(os.path.join(args.snapshots, snapshot_name, 'session.pkl'))

    return args, snapshot_name


def init_data_processing():
    kvs = GlobalKVS()
    train_augs = init_train_augs()

    dataset = OAProgressionDataset(dataset=kvs['args'].dataset_root, split=kvs['metadata'], transforms=train_augs)

    mean_vector, std_vector = init_mean_std(snapshots_dir=kvs['args'].snapshots,
                                            dataset=dataset, batch_size=kvs['args'].bs,
                                            n_threads=kvs['args'].n_threads)

    norm_trf = tv_transforms.Normalize(torch.from_numpy(mean_vector).float(),
                                       torch.from_numpy(std_vector).float())
    train_trf = tv_transforms.Compose([
        train_augs,
        partial(apply_by_index, transform=norm_trf, idx=0)
    ])

    val_trf = tv_transforms.Compose([
        img_labels2solt,
        slt.CropTransform(crop_size=(300, 300), crop_mode='c'),
        unpack_solt_data,
        partial(apply_by_index, transform=tv_transforms.ToTensor(), idx=0),
        partial(apply_by_index, transform=norm_trf, idx=0)
    ])

    kvs.update('train_trf', train_trf)
    kvs.update('val_trf', val_trf)
    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))


def init_mean_std(snapshots_dir, dataset, batch_size, n_threads):
    if os.path.isfile(os.path.join(snapshots_dir, 'mean_std.npy')):
        tmp = np.load(os.path.join(snapshots_dir, 'mean_std.npy'))
        mean_vector, std_vector = tmp
    else:
        tmp_loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_threads)
        mean_vector = None
        std_vector = None
        print(colored('==> ', 'green') + 'Calculating mean and std')
        for batch in tqdm(tmp_loader, total=len(tmp_loader)):
            imgs = batch['img']
            if mean_vector is None:
                mean_vector = np.zeros(imgs.size(1))
                std_vector = np.zeros(imgs.size(1))
            for j in range(mean_vector.shape[0]):
                mean_vector[j] += imgs[:, j, :, :].mean()
                std_vector[j] += imgs[:, j, :, :].std()

        mean_vector /= len(tmp_loader)
        std_vector /= len(tmp_loader)
        np.save(os.path.join(snapshots_dir, 'mean_std.npy'),
                [mean_vector.astype(np.float32), std_vector.astype(np.float32)])

    return mean_vector, std_vector


def init_loaders(x_train, x_val):
    kvs = GlobalKVS()
    train_dataset = OAProgressionDataset(dataset=kvs['args'].dataset_root,
                                         split=x_train,
                                         transforms=kvs['train_trf'])

    val_dataset = OAProgressionDataset(dataset=kvs['args'].dataset_root,
                                       split=x_val,
                                       transforms=kvs['val_trf'])

    train_loader = DataLoader(train_dataset, batch_size=kvs['args'].bs,
                              num_workers=kvs['args'].n_threads, shuffle=True,
                              drop_last=True,
                              worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))

    val_loader = DataLoader(val_dataset, batch_size=kvs['args'].val_bs,
                            num_workers=kvs['args'].n_threads)

    return train_loader, val_loader