import operator
import os
import time
from functools import partial

import numpy as np
import solt.core as slc
import solt.transforms as slt
import torch
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.utils.data import DataLoader
from torchvision import transforms as tv_transforms
from tqdm import tqdm

from oaprogression.kvs import GlobalKVS, git_info
from oaprogression.training.args import parse_args
from oaprogression.training.dataset import OAProgressionDataset, AgeSexBMIDataset
from oaprogression.training.dataset import init_train_augs, apply_by_index, img_labels2solt, unpack_solt_data


def init_snapshot_dir(snapshot_name_prefix=None):
    kvs = GlobalKVS()
    snapshot_name = time.strftime('%Y_%m_%d_%H_%M')
    if snapshot_name_prefix is not None:
        snapshot_name = f'{snapshot_name_prefix}_{snapshot_name}'
    os.makedirs(os.path.join(kvs['args'].snapshots, snapshot_name), exist_ok=True)
    kvs.update('snapshot_name', snapshot_name)


def init_session(snapshot_name_prefix=None):
    if not torch.cuda.is_available():
        raise EnvironmentError('The code must be run on GPU.')

    kvs = GlobalKVS()

    # Getting the arguments
    args = parse_args()
    kvs.update('args', args)

    # Initializing the seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Creating the snapshot
    init_snapshot_dir(snapshot_name_prefix)

    kvs.update('pytorch_version', torch.__version__)

    if torch.cuda.is_available():
        kvs.update('cuda', torch.version.cuda)
        kvs.update('gpus', torch.cuda.device_count())
    else:
        kvs.update('cuda', None)
        kvs.update('gpus', None)

    kvs.save_pkl(os.path.join(args.snapshots, kvs['snapshot_name'], 'session.pkl'))

    return args, kvs['snapshot_name']


def init_data_processing():
    kvs = GlobalKVS()
    train_augs = init_train_augs()

    dataset = OAProgressionDataset(dataset=kvs['args'].dataset_root, split=kvs['metadata'], trf=train_augs)

    mean_vector, std_vector = init_mean_std(snapshots_dir=kvs['args'].snapshots,
                                            dataset=dataset, batch_size=kvs['args'].bs,
                                            n_threads=kvs['args'].n_threads)

    print(colored('====> ', 'red') + 'Mean:', mean_vector)
    print(colored('====> ', 'red') + 'Std:', std_vector)

    norm_trf = tv_transforms.Normalize(mean_vector.tolist(), std_vector.tolist())
    train_trf = tv_transforms.Compose([
        train_augs,
        partial(apply_by_index, transform=norm_trf, idx=0)
    ])

    val_trf = tv_transforms.Compose([
        img_labels2solt,
        slc.Stream([
            slt.ResizeTransform((310, 310)),
            slt.CropTransform(crop_size=(300, 300), crop_mode='c'),
            slt.ImageColorTransform(mode='gs2rgb'),
        ], interpolation='bicubic'),
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


def init_loaders(x_train, x_val, progression=True):
    kvs = GlobalKVS()

    ds = OAProgressionDataset
    if not progression:
        ds = AgeSexBMIDataset

    train_dataset = ds(dataset=kvs['args'].dataset_root,
                       split=x_train,
                       trf=kvs['train_trf'])

    val_dataset = ds(dataset=kvs['args'].dataset_root,
                     split=x_val,
                     trf=kvs['val_trf'])

    train_loader = DataLoader(train_dataset, batch_size=kvs['args'].bs,
                              num_workers=kvs['args'].n_threads,
                              drop_last=True, shuffle=True,
                              worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))

    val_loader = DataLoader(val_dataset, batch_size=kvs['args'].val_bs,
                            num_workers=kvs['args'].n_threads)

    return train_loader, val_loader


def init_folds(project='OA_progression'):
    kvs = GlobalKVS()
    writers = {}
    cv_split_train = {}
    for fold_id, split in enumerate(kvs['cv_split_all_folds']):
        if kvs['args'].fold != -1 and fold_id != kvs['args'].fold:
            continue
        kvs.update(f'losses_fold_[{fold_id}]', None, list)
        kvs.update(f'val_metrics_fold_[{fold_id}]', None, list)
        cv_split_train[fold_id] = split
        writers[fold_id] = SummaryWriter(os.path.join(kvs['args'].logs,
                                                      project,
                                                      'fold_{}'.format(fold_id), kvs['snapshot_name']))

    kvs.update('cv_split_train', cv_split_train)
    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))
    return writers


def save_checkpoint(model, val_metric_name, comparator='lt'):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    kvs = GlobalKVS()
    fold_id = kvs['cur_fold']
    epoch = kvs['cur_epoch']
    val_metric = kvs[f'val_metrics_fold_[{fold_id}]'][-1][0][val_metric_name]
    comparator = getattr(operator, comparator)
    cur_snapshot_name = os.path.join(kvs['args'].snapshots, kvs['snapshot_name'],
                                     f'fold_{fold_id}_epoch_{epoch + 1}.pth')

    if kvs['prev_model'] is None:
        print(colored('====> ', 'red') + 'Snapshot was saved to', cur_snapshot_name)
        torch.save(model.state_dict(), cur_snapshot_name)
        kvs.update('prev_model', cur_snapshot_name)
        kvs.update('best_val_metric', val_metric)

    else:
        if comparator(val_metric, kvs['best_val_metric']):
            print(colored('====> ', 'red') + 'Snapshot was saved to', cur_snapshot_name)
            os.remove(kvs['prev_model'])
            torch.save(model.state_dict(), cur_snapshot_name)
            kvs.update('prev_model', cur_snapshot_name)
            kvs.update('best_val_metric', val_metric)

    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))
