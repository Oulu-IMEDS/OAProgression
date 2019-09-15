import cv2

from oaprogression.kvs import GlobalKVS
from oaprogression.training import dataset
from oaprogression.training import session
from oaprogression.training import train_utils

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

if __name__ == "__main__":
    kvs = GlobalKVS()
    session.init_session()
    dataset.init_progression_metadata()
    session.init_data_processing()
    writers = session.init_folds()
    train_utils.train_folds(writers)
