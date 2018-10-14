import cv2
from oaprogression.training.session import init_session

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

if __name__ == "__main__":
    args = init_session()
    print(args)