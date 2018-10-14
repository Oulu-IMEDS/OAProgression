import cv2
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

