import sys

from core.my_lblonly import My_lblonly

sys.path.append("..") # Adds higher directory to python modules path.

import warnings
warnings.filterwarnings("ignore")

from bboxes.rfbb import RandomForestBlackBox
from core import *
from core.original_lblonly import Original_lblonly
from multiprocessing import Process

# noise_samples_list = [1, 10, 50, 100, 200, 500, 1000, 5000]
noise_samples_list = [1, 10]

bb = RandomForestBlackBox()
result = []


def worker(noise_samples):
    # att = Original_lblonly(bb, noise_samples)
    att = My_lblonly(bb, 8, noise_samples)
    att.start_attack()


if __name__ == "__main__":
    processes = []
    for nise_num in noise_samples_list:
        process = Process(target=worker, args=(nise_num,))
        processes.append(process)
        process.start()

    for p in processes:
        p.join()