import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
import solt.transforms as slt
import solt.core as slc
import solt.data as sld
import copy
from functools import partial

from oaprogression.training import dataset
from oaprogression.kvs import GlobalKVS


def img_labels2solt(inp):
    img, KL, prog_increase, prog_label = inp
    return sld.DataContainer((img, KL, prog_increase, prog_label), fmt='ILLL')


def unpack_solt_data(dc: sld.DataContainer):
    return dc.data


def apply_by_index(items, transform, idx=0):
    """Applies callable to certain objects in iterable using given indices.
    Parameters
    ----------
    items: tuple or list
    transform: callable
    idx: int or tuple or or list None
    Returns
    -------
    result: tuple
    """
    if idx is None:
        return items
    if not isinstance(items, (tuple, list)):
        raise TypeError
    if not isinstance(idx, (int, tuple, list)):
        raise TypeError

    if isinstance(idx, int):
        idx = [idx, ]

    idx = set(idx)
    res = []
    for i, item in enumerate(items):
        if i in idx:
            res.append(transform(item))
        else:
            res.append(copy.deepcopy(item))

    return res


def init_train_augs():
    trf = transforms.Compose([
        img_labels2solt,
        slc.Stream([
            slt.ResizeTransform((310, 310)),
            slt.ImageAdditiveGaussianNoise(p=0.5, gain_range=0.3),
            slt.RandomRotate(p=1, rotation_range=(-10, 10)),
            slt.CropTransform(crop_size=(300, 300), crop_mode='r'),
            slt.ImageGammaCorrection(p=0.5, gamma_range=(0.5, 2)),
            slt.ImageColorTransform(mode='gs2rgb')
        ], interpolation='bicubic', padding='z'),
        unpack_solt_data,
        partial(apply_by_index, transform=transforms.ToTensor(), idx=0),
    ])
    return trf


def debug_augmentations(n_iter=20):
    kvs = GlobalKVS()

    ds = dataset.OAProgressionDataset(dataset=kvs['args'].dataset_root,
                                      split=kvs['metadata'],
                                      transforms=init_train_augs())

    for ind in np.random.choice(len(ds), n_iter, replace=False):
        sample = ds[ind]
        img = np.clip(sample['img'].numpy()*255, 0, 255).astype(np.uint8)
        img = np.swapaxes(img, 0, -1)
        img = np.swapaxes(img, 0, 1)
        plt.figure()
        plt.imshow(img)
        plt.show()



