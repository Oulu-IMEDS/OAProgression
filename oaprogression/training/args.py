import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='')
    parser.add_argument('--metadata_root', default='')
    parser.add_argument('--snapshots', default='')
    parser.add_argument('--logs', default='')
    parser.add_argument('--backbone', type=str, choices=['se_resnet50',
                                                         'inceptionv4',
                                                         'se_resnext50_32x4d',
                                                         'se_resnext101_32x4d'
                                                         ], default='se_resnext50_32x4d')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--val_bs', type=int, default=128)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--n_epochs', type=int, default=25)
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--loss_weight', type=float, default=0.5)
    parser.add_argument('--unfreeze_epoch', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_drop', default=[10, 15, 20])
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=123456)
    args = parser.parse_args()

    return args
