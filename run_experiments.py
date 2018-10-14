import os
import cv2
import numpy as np

from torch.optim.lr_scheduler import MultiStepLR

from termcolor import colored
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix, mean_squared_error

from oaprogression.kvs import GlobalKVS
from oaprogression.training import session
from oaprogression.training import train_utils
from oaprogression.training import dataset

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

if __name__ == "__main__":
    kvs = GlobalKVS()
    session.init_session()
    dataset.init_metadata()

    print(colored("==> ", 'green') + f"Combined dataset has "
                                     f"{(kvs['metadata'].Progressor == 0).sum()} non-progressed knees")
    print(colored("==> ", 'green')+f"Combined dataset has "
                                   f"{(kvs['metadata'].Progressor > 0).sum()} progressed knees")

    session.init_data_processing()

    print(colored('==> ', 'green') + 'Initialized the datasplits....')
    for fold_id, (train_index, val_index) in enumerate(kvs['cv_split']):
        print(colored('====> ', 'blue') + f'Training fold {fold_id}....')
        if kvs['args'].fold != -1 and fold_id != kvs['args'].fold:
            continue

        train_loader, val_loader = session.init_loaders(kvs['metadata'].iloc[train_index],
                                                        kvs['metadata'].iloc[val_index])

        net = train_utils.init_model()
        optimizer = train_utils.init_optimizer([{'params': net.module.classifier_kl.parameters()},
                                                {'params': net.module.classifier_prog.parameters()}])
        scheduler = MultiStepLR(optimizer, milestones=kvs['args'].lr_drop, gamma=0.1)

        writer = SummaryWriter(os.path.join(kvs['args'].logs,
                                            'OA_progression', 'fold_{}'.format(fold_id),
                                            kvs['snapshot_name']))

        for epoch in range(kvs['args'].n_epochs):
            if epoch == kvs['args'].unfreeze_epoch:
                print(colored('==> ', 'red')+'Unfreezing the layers!')
                new_lr_drop_milestones = list(map(lambda x: x-kvs['args'].unfreeze_epoch, kvs['args'].lr_drop))
                optimizer = train_utils.init_optimizer(net.parameters())
                scheduler = MultiStepLR(optimizer, milestones=new_lr_drop_milestones, gamma=0.1)

            train_loss = train_utils.train_epoch(epoch, net, optimizer, train_loader)
            val_out = train_utils.validate_epoch(epoch, net, val_loader)
            val_loss, val_ids, gt_progression, preds_progression, gt_kl, preds_kl = val_out

            # Computing Validation metrics
            preds_progression_bin = preds_progression[:, 1:].sum(1)
            preds_kl_bin = preds_kl[:, 1:].sum(1)

            cm_prog = confusion_matrix(gt_progression, preds_progression.argmax(1))
            cm_kl = confusion_matrix(gt_kl, preds_kl.argmax(1))

            kappa_prog = cohen_kappa_score(gt_progression, preds_progression.argmax(1), weights="quadratic")
            acc_prog = np.mean(cm_prog.diagonal().astype(float) / cm_prog.sum(axis=1))
            mse_prog = mean_squared_error(gt_progression, preds_progression.argmax(1))
            auc_prog = roc_auc_score(gt_progression > 0, preds_progression_bin)

            kappa_kl = cohen_kappa_score(gt_kl, preds_kl.argmax(1), weights="quadratic")
            acc_kl = np.mean(cm_kl.diagonal().astype(float) / cm_kl.sum(axis=1))
            mse_kl = mean_squared_error(gt_kl, preds_kl.argmax(1))
            auc_oa = roc_auc_score(gt_kl > 1, preds_kl_bin)

            print(colored('====> ', 'green')+f'Val. loss: {val_loss:.5f}')
            print(colored('====> ', 'green')+f'AUC [prog]: {auc_prog:.5f}')
            print(colored('====> ', 'green')+f'Kappa [prog]: {kappa_prog:.5f}')
            print(colored('====> ', 'green')+f'MSE [prog]: {mse_prog:.5f}')
            print(colored('====> ', 'green')+f'Accuracy [prog]: {acc_prog:.5f}')

            writer.add_scalars('Losses', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('Kappas progression', {'val': kappa_prog}, epoch)
            writer.add_scalars('Acc progression', {'val': acc_prog}, epoch)
            writer.add_scalars('AUC progression', {'val': auc_prog}, epoch)
            scheduler.step()
