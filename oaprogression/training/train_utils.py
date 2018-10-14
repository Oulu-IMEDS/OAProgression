from torch import nn
from torch import optim
from oaprogression.kvs import GlobalKVS
from oaprogression.training.model import KneeNet


def init_model():
    kvs = GlobalKVS()
    net = KneeNet(kvs['args'].backbone, kvs['args'].dropout_rate)

    if kvs['gpus'] > 1:
        net = nn.DataParallel(net).to('cuda')

    net = net.to('cuda')
    return net


def init_optimizer(net):
    kvs = GlobalKVS()
    if kvs['args'].optimizer == 'adam':
        return optim.Adam(net.parameters(), lr=kvs['args'].lr, weight_decay=kvs['args'].wd)
    elif kvs['args'].optimizer == 'sgd':
        return optim.SGD(net.parameters(), lr=kvs['args'].lr, weight_decay=kvs['args'].wd, momentum=0.9)
    else:
        raise NotImplementedError
