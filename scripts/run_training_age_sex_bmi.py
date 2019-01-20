import sys

import cv2

from torch.optim.lr_scheduler import MultiStepLR

from termcolor import colored

from oaprogression.kvs import GlobalKVS
from oaprogression.training import session
from oaprogression.training import train_utils
from oaprogression.training import dataset

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


DEBUG = sys.gettrace() is not None

if __name__ == "__main__":
    kvs = GlobalKVS()
    session.init_session()
    dataset.init_age_sex_bmi_metadata()
    session.init_data_processing()
    writers = session.init_folds(kvs['args'].target_var + '_prediction')

    if DEBUG:
        dataset.debug_augmentations()

    print(kvs['metadata'].AGE.min(), kvs['metadata'].AGE.max())
    print(kvs['metadata'].BMI.min(), kvs['metadata'].BMI.max())

    print(kvs['metadata'].AGE.mean(), kvs['metadata'].AGE.std())
    print(kvs['metadata'].BMI.mean(), kvs['metadata'].BMI.std())

    for fold_id in kvs['cv_split_train']:
        kvs.update('cur_fold', fold_id)
        kvs.update('prev_model', None)
        print(colored('====> ', 'blue') + f'Training fold {fold_id}....')

        train_index, val_index = kvs['cv_split_train'][fold_id]
        train_loader, val_loader = session.init_loaders(kvs['metadata'].iloc[train_index],
                                                        kvs['metadata'].iloc[val_index], progression=False)

        net = train_utils.init_model(kneenet=False)
        optimizer = train_utils.init_optimizer([{'params': net.module.fc.parameters()}, ])

        scheduler = MultiStepLR(optimizer, milestones=kvs['args'].lr_drop, gamma=0.1)

        for epoch in range(kvs['args'].n_epochs):
            kvs.update('cur_epoch', epoch)
            if epoch == kvs['args'].unfreeze_epoch:
                print(colored('====> ', 'red')+'Unfreezing the layers!')
                new_lr_drop_milestones = list(map(lambda x: x-kvs['args'].unfreeze_epoch, kvs['args'].lr_drop))
                optimizer.add_param_group({'params': net.module.encoder.parameters()})
                scheduler = MultiStepLR(optimizer, milestones=new_lr_drop_milestones, gamma=0.1)

            print(colored('====> ', 'red') + 'LR:', scheduler.get_lr())
            train_loss = train_utils.epoch_pass(net, optimizer, train_loader)
            val_loss, val_ids, gt, preds = train_utils.epoch_pass(net, None, val_loader)
            train_utils.log_metrics_age_sex_bmi(writers[fold_id], train_loss, val_loss, gt, preds)

            session.save_checkpoint(net, 'val_loss', 'lt')
            scheduler.step()
