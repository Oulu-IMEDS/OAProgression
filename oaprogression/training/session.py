import os
import time
import numpy as np
from tqdm import tqdm
from termcolor import colored
import torch
from torch import optim
from torch.utils.data import DataLoader
import pandas as pd
from oaprogression.training.args import parse_args
from oaprogression.training.dataset import OAProgressionDataset
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


def init_optimizer(net):
    kvs = GlobalKVS()
    if kvs['args'].optimizer == 'adam':
        return optim.Adam(net.parameters(), lr=kvs['args'].lr, weight_decay=kvs['args'].wd)
    elif kvs['args'].optimizer == 'sgd':
        return optim.SGD(net.parameters(), lr=kvs['args'].lr, weight_decay=kvs['args'].wd, momentum=0.9)
    else:
        raise NotImplementedError

"""
def init_data_processing():
    kvs = GlobalKVS()
    train_augs = build_train_augmentation_pipeline()

    dataset = OAProgressionDataset(split=kvs['metadata'],
                                  trf=train_augs,
                                  read_img=read_gs_ocv,
                                  read_mask=read_gs_mask_ocv)

    mean_vector, std_vector, class_weights = init_mean_std(snapshots_dir=kvs['args'].snapshots,
                                                           dataset=dataset,
                                                           batch_size=kvs['args'].bs,
                                                           n_threads=kvs['args'].n_threads,
                                                           n_classes=kvs['args'].n_classes)

    norm_trf = transforms.Normalize(torch.from_numpy(mean_vector).float(),
                                    torch.from_numpy(std_vector).float())
    train_trf = transforms.Compose([
        train_augs,
        partial(apply_by_index, transform=norm_trf, idx=0)
    ])

    val_trf = transforms.Compose([
        partial(apply_by_index, transform=gs2tens, idx=[0, 1]),
        partial(apply_by_index, transform=norm_trf, idx=0)
    ])
    kvs.update('class_weights', class_weights)
    kvs.update('train_trf', train_trf)
    kvs.update('val_trf', val_trf)
    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))
"""


def init_mean_std(snapshots_dir, dataset, batch_size, n_threads, n_classes):
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

