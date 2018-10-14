import cv2
from termcolor import colored
from oaprogression.kvs import GlobalKVS
from oaprogression.training import session
from oaprogression.training import dataset


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

if __name__ == "__main__":
    kvs = GlobalKVS()
    session.init_session()
    dataset.init_metadata()
    session.init_data_processing()

    print(colored("Combined dataset has ", 'green') + f"{(kvs['metadata'].Progressor == 0).sum()} non-progressed knees")
    print(colored("Combined dataset has ", 'green')+f"{(kvs['metadata'].Progressor > 0).sum()} progressed knees")
