import gc
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, roc_auc_score, median_absolute_error
from termcolor import colored
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
from torch import optim
from tqdm import tqdm

from oaprogression.evaluation import tools as testtools
from oaprogression.kvs import GlobalKVS
from oaprogression.training.model import KneeNet, PretrainedModel
from oaprogression.training import session


def train_folds(writers):
    kvs = GlobalKVS()
    for fold_id in kvs['cv_split_train']:
        kvs.update('cur_fold', fold_id)
        kvs.update('prev_model', None)
        print(colored('====> ', 'blue') + f'Training fold {fold_id}....')

        train_index, val_index = kvs['cv_split_train'][fold_id]
        train_loader, val_loader = session.init_loaders(kvs['metadata'].iloc[train_index],
                                                        kvs['metadata'].iloc[val_index])

        net = init_model()
        optimizer = init_optimizer([{'params': net.module.classifier_kl.parameters()},
                                                {'params': net.module.classifier_prog.parameters()}])

        scheduler = MultiStepLR(optimizer, milestones=kvs['args'].lr_drop, gamma=0.1)

        for epoch in range(kvs['args'].n_epochs):
            kvs.update('cur_epoch', epoch)
            if epoch == kvs['args'].unfreeze_epoch:
                print(colored('====> ', 'red') + 'Unfreezing the layers!')
                new_lr_drop_milestones = list(map(lambda x: x - kvs['args'].unfreeze_epoch, kvs['args'].lr_drop))
                optimizer.add_param_group({'params': net.module.features.parameters()})
                scheduler = MultiStepLR(optimizer, milestones=new_lr_drop_milestones, gamma=0.1)

            print(colored('====> ', 'red') + 'LR:', scheduler.get_lr())
            train_loss = prog_epoch_pass(net, optimizer, train_loader)
            val_out = prog_epoch_pass(net, None, val_loader)
            val_loss, val_ids, gt_progression, preds_progression, gt_kl, preds_kl = val_out
            log_metrics_prog(writers[fold_id], train_loss, val_loss,
                                         gt_progression, preds_progression, gt_kl, preds_kl)

            session.save_checkpoint(net, 'ap_prog', 'gt')
            scheduler.step()


def init_model(kneenet=True):
    kvs = GlobalKVS()
    if kneenet:
        net = KneeNet(kvs['args'].backbone, kvs['args'].dropout_rate)
    else:
        if not kvs['args'].predict_age_sex_bmi:
            net = PretrainedModel(kvs['args'].backbone, kvs['args'].dropout_rate, 1, True)
        else:
            net = PretrainedModel(kvs['args'].backbone, kvs['args'].dropout_rate, 3, True)

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
    # Individual factors prediction
    preds = list()
    gt = list()
    # Predicting Age, Sex, BMI
    preds_age = list()
    preds_sex = list()
    preds_bmi = list()

    gt_age = list()
    gt_sex = list()
    gt_bmi = list()

    ids = list()

    with torch.set_grad_enabled(optimizer is not None):
        for i, batch in enumerate(loader):
            if optimizer is not None:
                optimizer.zero_grad()
            inp = batch['img'].to(device)
            output = net(inp).squeeze()
            if not kvs['args'].predict_age_sex_bmi:
                target = batch[kvs['args'].target_var].float().to(device)
                loss = criterion(output, target)
            else:
                target_age = batch['AGE'].float().to(device)
                target_sex = batch['SEX'].float().to(device)
                target_bmi = batch['BMI'].float().to(device)
                loss_age = F.mse_loss(output[:, 0].squeeze(), target_age)
                loss_sex = F.binary_cross_entropy_with_logits(output[:, 1].squeeze(), target_sex)
                loss_bmi = F.mse_loss(output[:, 2].squeeze(), target_bmi)
                loss = loss_age + loss_sex + loss_bmi

            if optimizer is not None:
                loss.backward()
                if kvs['args'].clip_grad:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), kvs['args'].clip_grad_norm)
                optimizer.step()
            else:
                if not kvs['args'].predict_age_sex_bmi:
                    if kvs['args'].target_var == 'SEX':
                        pred_batch = torch.sigmoid(output).data.to('cpu').numpy().squeeze()
                    else:
                        pred_batch = output.data.to('cpu').numpy().squeeze()
                    preds.append(pred_batch)
                    gt.append(batch[kvs['args'].target_var].numpy().squeeze())
                else:
                    preds_age_batch = output[:, 0].data.to('cpu').numpy().squeeze()
                    preds_sex_batch = torch.sigmoid(output[:, 1]).data.to('cpu').numpy().squeeze()
                    preds_bmi_batch = output[:, 2].data.to('cpu').numpy().squeeze()

                    preds_age.append(preds_age_batch)
                    preds_sex.append(preds_sex_batch)
                    preds_bmi.append(preds_bmi_batch)

                    gt_age.append(batch['AGE'].numpy().squeeze())
                    gt_sex.append(batch['SEX'].numpy().squeeze())
                    gt_bmi.append(batch['BMI'].numpy().squeeze())

                ids.extend(batch['ID_SIDE'])

            running_loss += loss.item()
            if optimizer is not None:
                pbar.set_description(f'Training   [{epoch} / {max_epoch}]:: {running_loss / (i + 1):.3f}')
            else:
                pbar.set_description(f'Validating [{epoch} / {max_epoch}]:')
            pbar.update()

            gc.collect()

    gc.collect()
    pbar.close()

    if optimizer is not None:
        return running_loss / n_batches
    else:
        if not kvs['args'].predict_age_sex_bmi:
            preds = np.hstack(preds)
            gt = np.hstack(gt)
            return running_loss / n_batches, ids, gt, preds
        else:
            preds_age = np.hstack(preds_age)
            gt_age = np.hstack(gt_age)

            preds_sex = np.hstack(preds_sex)
            gt_sex = np.hstack(gt_sex)

            preds_bmi = np.hstack(preds_bmi)
            gt_bmi = np.hstack(gt_bmi)

            return running_loss / n_batches, ids, gt_age, preds_age, gt_sex, preds_sex, gt_bmi, preds_bmi


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
        return running_loss / n_batches, ids, gt_progression, preds_progression, gt_kl, preds_kl


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


def log_metrics_age_sex_bmi(boardlogger, train_loss, val_res):
    kvs = GlobalKVS()
    res = dict()
    val_loss = val_res[0]
    res['val_loss'] = val_loss,
    res['epoch'] = kvs['cur_epoch']
    print(colored('====> ', 'green') + f'Train loss: {train_loss:.5f}')
    print(colored('====> ', 'green') + f'Validation loss: {val_loss:.5f}')
    boardlogger.add_scalars('Losses', {'train': train_loss, 'val': val_loss}, kvs['cur_epoch'])

    if not kvs['args'].predict_age_sex_bmi:
        _, ids, gt, preds = val_res
        if kvs['args'].target_var == 'SEX':
            val_auc = roc_auc_score(gt.astype(int), preds)
            res['sex_auc'] = val_auc
            print(colored('====> ', 'green') + f'Validation AUC: {val_auc:.5f}')
            boardlogger.add_scalars('AUC sex', {'val': res['sex_auc']}, kvs['cur_epoch'])
        else:
            val_mse = mean_squared_error(gt, preds)
            val_mae = median_absolute_error(gt, preds)
            res[f"{kvs['args'].target_var}_mse"] = val_mse
            res[f"{kvs['args'].target_var}_mae"] = val_mae

            print(colored('====> ', 'green') + f'Validation mae: {val_mae:.5f}')
            print(colored('====> ', 'green') + f'Validation mse: {val_mse:.5f}')

            boardlogger.add_scalars(f"MSE [{kvs['args'].target_var}]", {'val': val_mse},
                                    kvs['cur_epoch'])
            boardlogger.add_scalars(f"MAE [{kvs['args'].target_var}]", {'val': val_mae},
                                    kvs['cur_epoch'])
    else:
        _, ids, gt_age, preds_age, gt_sex, preds_sex, gt_bmi, preds_bmi = val_res
        val_mse_age = mean_squared_error(gt_age, preds_age)
        val_mae_age = median_absolute_error(gt_age, preds_age)
        val_sex_auc = roc_auc_score(gt_sex.astype(int), preds_sex)
        val_mse_bmi = mean_squared_error(gt_bmi, preds_bmi)
        val_mae_bmi = median_absolute_error(gt_bmi, preds_bmi)

        res["AGE_mse"] = val_mse_age
        res["AGE_mae"] = val_mae_age

        res["BMI_mse"] = val_mse_bmi
        res["BMI_mae"] = val_mae_bmi

        res["SEX_auc"] = val_sex_auc

        print(colored('====> ', 'green') + f'Validation mae [Age]: {val_mae_age:.5f}')
        print(colored('====> ', 'green') + f'Validation mse [Age]: {val_mse_age:.5f}')

        print(colored('====> ', 'green') + f'Validation val_auc [Sex]: {val_sex_auc:.5f}')

        print(colored('====> ', 'green') + f'Validation mae [BMI]: {val_mae_bmi:.5f}')
        print(colored('====> ', 'green') + f'Validation mse [BMI]: {val_mse_bmi:.5f}')

    kvs.update(f'losses_fold_[{kvs["cur_fold"]}]', {'epoch': kvs['cur_epoch'],
                                                    'train_loss': train_loss,
                                                    'val_loss': val_loss})

    kvs.update(f'val_metrics_fold_[{kvs["cur_fold"]}]', res)

    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['snapshot_name'], 'session.pkl'))
