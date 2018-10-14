import os
import cv2

from torch.optim.lr_scheduler import MultiStepLR

from termcolor import colored
from tensorboardX import SummaryWriter
from sklearn.model_selection import GroupKFold

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
    print(colored("==> ",'green')+f"Combined dataset has "
                                  f"{(kvs['metadata'].Progressor > 0).sum()} progressed knees")

    session.init_data_processing()

    gkf = GroupKFold(n_splits=5)
    cv_split = gkf.split(kvs['metadata'], kvs['metadata']['Progressor'], kvs['metadata']['ID'].astype(str))
    print(colored('==> ', 'green') + 'Initialized the datasplits....')
    for fold_id, (train_index, val_index) in enumerate(cv_split):
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
                optimizer = train_utils.init_optimizer(net)
                scheduler = MultiStepLR(optimizer, milestones=new_lr_drop_milestones, gamma=0.1)
            train_loss = train_utils.train_epoch(epoch, net, optimizer, train_loader)
