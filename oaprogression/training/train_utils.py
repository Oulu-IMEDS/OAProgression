from tqdm import tqdm
import gc
import numpy as np

import torch
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


def init_optimizer(parameters):
    kvs = GlobalKVS()
    if kvs['args'].optimizer == 'adam':
        return optim.Adam(parameters, lr=kvs['args'].lr, weight_decay=kvs['args'].wd)
    elif kvs['args'].optimizer == 'sgd':
        return optim.SGD(parameters, lr=kvs['args'].lr, weight_decay=kvs['args'].wd, momentum=0.9)
    else:
        raise NotImplementedError


def train_epoch(net, optimizer, train_loader):
    kvs = GlobalKVS()
    net.train(True)
    running_loss = 0.0
    n_batches = len(train_loader)
    pbar = tqdm(total=len(train_loader))
    epoch = kvs['cur_epoch']
    max_epoch = kvs['args'].n_epochs
    device = next(net.parameters()).device
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        # forward + backward + optimize
        labels_prog = batch['label'].long().to(device)
        labels_kl = batch['KL'].long().to(device)
        inputs = batch['img'].to(device)

        outputs_kl, outputs_prog = net(inputs)

        loss_kl = F.cross_entropy(outputs_kl, labels_kl)
        loss_prog = F.cross_entropy(outputs_prog, labels_prog)
        loss = loss_prog.mul(kvs['args'].loss_weight) + loss_kl.mul(1 - kvs['args'].loss_weight)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_description(f'Training [{epoch} / {max_epoch}]:: {running_loss / (i + 1):.3f}')
        pbar.update()

        gc.collect()
    gc.collect()
    pbar.close()
    return running_loss / n_batches


def validate_epoch(net, val_loader):
    kvs = GlobalKVS()
    net.eval()
    running_loss = 0.0
    n_batches = len(val_loader)
    epoch = kvs['cur_epoch']
    max_epoch = kvs['args'].n_epochs
    preds_progression = []
    gt_progression = []

    preds_kl = []
    gt_kl = []
    device = next(net.parameters()).device
    ids = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader),
                             desc=f'Validating [{epoch} / {max_epoch}]:: '):
            labels_prog = batch['label'].long().to(device)
            labels_kl = batch['KL'].long().to(device)
            inputs = batch['img'].to(device)

            outputs_kl, outputs_prog = net(inputs)

            probs_progression_batch = F.softmax(outputs_prog, 1).data.to('cpu').numpy()
            probs_kl_batch = F.softmax(outputs_kl, 1).data.to('cpu').numpy()

            loss_kl = F.cross_entropy(outputs_kl, labels_kl)
            loss_prog = F.cross_entropy(outputs_prog, labels_prog)
            loss = loss_prog.mul(kvs['args'].loss_weight) + loss_kl.mul(1 - kvs['args'].loss_weight)

            preds_progression.append(probs_progression_batch)
            gt_progression.append(batch['label'].numpy())

            preds_kl.append(probs_kl_batch)
            gt_kl.append(batch['KL'].numpy())
            ids.extend(batch['ID_SIDE'])

            running_loss += loss.item()
            gc.collect()
        gc.collect()

    preds_progression = np.vstack(preds_progression)
    gt_progression = np.hstack(gt_progression)

    preds_kl = np.vstack(preds_kl)
    gt_kl = np.hstack(gt_kl)

    return running_loss/n_batches, ids, gt_progression, preds_progression, gt_kl, preds_kl