from tqdm import tqdm
import gc
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, median_absolute_error
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from oaprogression.kvs import GlobalKVS
from oaprogression.training.model import KneeNet, PretrainedModel

import os
from termcolor import colored
from oaprogression.evaluation import tools as testtools


def init_model(kneenet=True):
    kvs = GlobalKVS()
    if kneenet:
        net = KneeNet(kvs['args'].backbone, kvs['args'].dropout_rate)
    else:
        net = PretrainedModel(kvs['args'].backbone, kvs['args'].dropout_rate, 1, True)

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


def init_epoch_pass(net, optimizer, loader):
    kvs = GlobalKVS()
    net.train(optimizer is not None)
    running_loss = 0.0
    n_batches = len(loader)
    pbar = tqdm(total=n_batches)
    epoch = kvs['cur_epoch']
    max_epoch = kvs['args'].n_epochs
    device = next(net.parameters()).device
    return running_loss, pbar, n_batches, epoch, max_epoch, device


def epoch_pass(net, optimizer, loader):
    kvs = GlobalKVS()
    running_loss, pbar, n_batches, epoch, max_epoch, device = init_epoch_pass(net, optimizer, loader)
    if kvs['args'].target_var == 'SEX':
        criterion = F.binary_cross_entropy_with_logits
    else:
        criterion = F.mse_loss
    preds = []
    gt = []
    ids = []

    with torch.set_grad_enabled(optimizer is not None):
        for i, batch in enumerate(loader):
            if optimizer is not None:
                optimizer.zero_grad()
            inp = batch['img'].to(device)
            target = batch[kvs['args'].target_var].float().to(device)

            output = net(inp).squeeze()
            loss = criterion(output, target)

            if optimizer is not None:
                loss.backward()
                if kvs['args'].clip_grad:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), kvs['args'].clip_grad_norm)
                optimizer.step()
            else:
                if kvs['args'].target_var == 'SEX':
                    pred_batch = torch.sigmoid(output).data.to('cpu').numpy().squeeze()
                else:
                    pred_batch = output.data.to('cpu').numpy().squeeze()

                preds.append(pred_batch)
                gt.append(batch[kvs['args'].target_var].numpy().squeeze())
                ids.extend(batch['ID_SIDE'])

            running_loss += loss.item()
            if optimizer is not None:
                pbar.set_description(f'Training   [{epoch} / {max_epoch}]:: {running_loss / (i + 1):.3f}')
            else:
                pbar.set_description(f'Validating [{epoch} / {max_epoch}]:')
            pbar.update()

            gc.collect()

    if optimizer is None:
        preds = np.hstack(preds)
        gt = np.hstack(gt)

    gc.collect()
    pbar.close()

    if optimizer is not None:
        return running_loss / n_batches
    else:
        return running_loss/n_batches, ids, gt, preds


def prog_epoch_pass(net, optimizer, loader):
    kvs = GlobalKVS()
    running_loss, pbar, n_batches, epoch, max_epoch, device = init_epoch_pass(net, optimizer, loader)

    preds_progression = []
    gt_progression = []
    ids = []
    preds_kl = []
    gt_kl = []

    with torch.set_grad_enabled(optimizer is not None):
        for i, batch in enumerate(loader):
            if optimizer is not None:
                optimizer.zero_grad()
            # forward + backward + optimize if train
            labels_prog = batch['label'].long().to(device)
            labels_kl = batch['KL'].long().to(device)

            inputs = batch['img'].to(device)

            outputs_kl, outputs_prog = net(inputs)
            loss_kl = F.cross_entropy(outputs_kl, labels_kl)
            loss_prog = F.cross_entropy(outputs_prog, labels_prog)

            loss = loss_prog.mul(kvs['args'].loss_weight) + loss_kl.mul(1 - kvs['args'].loss_weight)

            if optimizer is not None:
                loss.backward()
                if kvs['args'].clip_grad:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), kvs['args'].clip_grad_norm)
                optimizer.step()
            else:
                probs_progression_batch = F.softmax(outputs_prog, 1).data.to('cpu').numpy()
                probs_kl_batch = F.softmax(outputs_kl, 1).data.to('cpu').numpy()

                preds_progression.append(probs_progression_batch)
                gt_progression.append(batch['label'].numpy())

                preds_kl.append(probs_kl_batch)
                gt_kl.append(batch['KL'])
                ids.extend(batch['ID_SIDE'])

            running_loss += loss.item()
            if optimizer is not None:
                pbar.set_description(f'Training   [{epoch} / {max_epoch}]:: {running_loss / (i + 1):.3f}')
            else:
                pbar.set_description(f'Validating [{epoch} / {max_epoch}]:')
            pbar.update()

            gc.collect()

    if optimizer is None:
        preds_progression = np.vstack(preds_progression)
        gt_progression = np.hstack(gt_progression)

        preds_kl = np.vstack(preds_kl)
        gt_kl = np.hstack(gt_kl)

    gc.collect()
    pbar.close()

    if optimizer is not None:
        return running_loss / n_batches
    else:
        return running_loss/n_batches, ids, gt_progression, preds_progression, gt_kl, preds_kl


def log_metrics_prog(boardlogger, train_loss, val_loss, gt_progression, preds_progression, gt_kl, preds_kl):
    kvs = GlobalKVS()

    res = testtools.calc_metrics(gt_progression, gt_kl, preds_progression, preds_kl)
    res['val_loss'] = val_loss,
    res['epoch'] = kvs['cur_epoch']

    print(colored('====> ', 'green') + f'Train loss: {train_loss:.5f}')
    print(colored('====> ', 'green') + f'Validation loss: {val_loss:.5f}')
    print(colored('====> ', 'green') + f'Validation AUC [prog]: {res["auc_prog"]:.5f}')
    print(colored('====> ', 'green') + f'Validation F1 @ 0.3 [prog]: {res["f1_score_03_prog"]:.5f}')
    print(colored('====> ', 'green') + f'Validation F1 @ 0.4 [prog]: {res["f1_score_04_prog"]:.5f}')
    print(colored('====> ', 'green') + f'Validation F1 @ 0.5 [prog]: {res["f1_score_05_prog"]:.5f}')
    print(colored('====> ', 'green') + f'Validation AP [prog]: {res["ap_prog"]:.5f}')

    print(colored('====> ', 'green') + f'Validation AUC [oa]: {res["auc_oa"]:.5f}')
    print(colored('====> ', 'green') + f'Kappa [oa]: {res["kappa_kl"]:.5f}')

    boardlogger.add_scalars('Losses', {'train': train_loss, 'val': val_loss}, kvs['cur_epoch'])
    boardlogger.add_scalars('AUC progression', {'val': res['auc_prog']}, kvs['cur_epoch'])
    boardlogger.add_scalars('F1-score @ 0.3 progression', {'val': res['f1_score_03_prog']}, kvs['cur_epoch'])
    boardlogger.add_scalars('F1-score @ 0.4 progression', {'val': res['f1_score_04_prog']}, kvs['cur_epoch'])
    boardlogger.add_scalars('F1-score @ 0.5 progression', {'val': res['f1_score_05_prog']}, kvs['cur_epoch'])
    boardlogger.add_scalars('Average Precision progression', {'val': res['ap_prog']}, kvs['cur_epoch'])

    kvs.update(f'losses_fold_[{kvs["cur_fold"]}]', {'epoch': kvs['cur_epoch'],
                                                    'train_loss': train_loss,
                                                    'val_loss': val_loss})

    kvs.update(f'val_metrics_fold_[{kvs["cur_fold"]}]', res)

    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))


def log_metrics_age_sex_bmi(boardlogger, train_loss, val_loss, gt, preds):
    kvs = GlobalKVS()
    val_auc, val_mse, val_mae = None, None, None
    res = dict()
    res['val_loss'] = val_loss,
    res['epoch'] = kvs['cur_epoch']
    if kvs['args'].target_var == 'SEX':
        val_auc = roc_auc_score(gt.astype(int), preds)
        res['sex_auc'] = val_auc
        boardlogger.add_scalars('AUC sex', {'val': res['sex_auc']}, kvs['cur_epoch'])
    else:
        val_mse = mean_squared_error(gt, preds)
        val_mae = median_absolute_error(gt, preds)
        res[f"{kvs['args'].target_var}_mse"] = val_mse
        res[f"{kvs['args'].target_var}_mae"] = val_mae

        boardlogger.add_scalars(f"MSE [{kvs['args'].target_var}]", {'val': val_mse},
                                kvs['cur_epoch'])
        boardlogger.add_scalars(f"MAE [{kvs['args'].target_var}]", {'val': val_mae},
                                kvs['cur_epoch'])

    print(colored('====> ', 'green') + f'Train loss: {train_loss:.5f}')
    print(colored('====> ', 'green') + f'Validation loss: {val_loss:.5f}')
    if kvs['args'].target_var == 'SEX':
        print(colored('====> ', 'green') + f'Validation AUC: {val_auc:.5f}')
    else:
        print(colored('====> ', 'green') + f'Validation mae: {val_mae:.5f}')
        print(colored('====> ', 'green') + f'Validation mse: {val_loss:.5f}')
    boardlogger.add_scalars('Losses', {'train': train_loss, 'val': val_loss}, kvs['cur_epoch'])

    kvs.update(f'losses_fold_[{kvs["cur_fold"]}]', {'epoch': kvs['cur_epoch'],
                                                    'train_loss': train_loss,
                                                    'val_loss': val_loss})

    kvs.update(f'val_metrics_fold_[{kvs["cur_fold"]}]', res)

    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))
