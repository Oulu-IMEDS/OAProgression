from torch import nn

from oaprogression.kvs import GlobalKVS
from oaprogression.training.model import KneeNet


def init_model():
    kvs = GlobalKVS()
    net = KneeNet(kvs['args'].backbone, kvs['args'].dropout_rate)

    if kvs['gpus'] > 1:
        net = nn.DataParallel(net).to('cuda')

    net = net.to('cuda')
    return net
