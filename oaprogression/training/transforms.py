from torchvision import transforms
import solt.transforms as slt
import solt.core as slc
import solt.data as sld
import copy
from functools import partial


def img_labels2solt(inp):
    img, KL, prog_label = inp
    return sld.DataContainer((img, KL, prog_label), fmt='ILL')


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
            slt.ImageAdditiveGaussianNoise(p=0.5, gain_range=0.2),
            slt.RandomRotate(rotation_range=(-5, 5)),
            slt.CropTransform(crop_size=(300, 300), crop_mode='r'),
            slt.ImageGammaCorrection(p=1, gamma_range=(0.7, 2)),
            slt.ImageColorTransform(mode='gs2rgb')
        ]),
        unpack_solt_data,
        partial(apply_by_index, transform=transforms.ToTensor(), idx=0),
    ])
    return trf