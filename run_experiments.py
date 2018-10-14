import cv2
from termcolor import colored
from oaprogression.kvs import GlobalKVS
from oaprogression.training.session import init_session
from oaprogression.training.dataset import init_metadata


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

if __name__ == "__main__":
    kvs = GlobalKVS()
    init_session()
    init_metadata()
    print(colored("Combined dataset has ", 'green') + f"{(kvs['metadata'].Progressor == 0).sum()} non-progressed knees")
    print(colored("Combined dataset has ", 'green')+f"{(kvs['metadata'].Progressor > 0).sum()} progressed knees")
