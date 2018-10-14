from tqdm import tqdm
import gc

from torch import nn
import torch.nn.functional as F
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


def train_epoch(epoch, net, optimizer, train_loader):
    kvs = GlobalKVS()
    net.train(True)
    running_loss = 0.0
    n_batches = len(train_loader)
    pbar = tqdm(total=len(train_loader))
    max_epoch = kvs['args'].n_epochs
    device = next(net.parameters()).device
    for i, sample in enumerate(train_loader):
        optimizer.zero_grad()
        # forward + backward + optimize
        labels_prog = sample['label'].long().to(device)
        labels_kl = sample['KL'].long().to(device)
        inputs = sample['img'].to(device)
        outputs_kl, outputs_prog = net(inputs)

        loss_kl = F.cross_entropy(outputs_kl, labels_kl)
        loss_prog = F.cross_entropy(outputs_prog, labels_prog)
        loss = loss_prog.mul(kvs['args'].loss_weight) + loss_kl.mul(1 - kvs['args'].loss_weight)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_description(f'[{epoch} / {max_epoch}] Run_loss {running_loss / (i + 1):.3f}')
        pbar.update()

        gc.collect()
    gc.collect()
    pbar.close()
    return running_loss / n_batches

