import sys

from core.confidence_attack import ConfidenceAttack

sys.path.append("..") # Adds higher directory to python modules path.

import warnings
warnings.filterwarnings("ignore")

from bboxes.rfbb import RandomForestBlackBox
from core import *
from multiprocessing import Process

noise_samples_list = [1, 10, 50, 100, 200, 500, 1000]
n_shadow_models = [1, 2, 4, 8, 16, 32]

bb = RandomForestBlackBox()
result = []


def worker(noise_models, noise_samples):
    att = ConfidenceAttack(bb, noise_models)
    # att = My_lblonly(bb, noise_models, noise_samples)
    att.start_attack()


if __name__ == "__main__":
    processes = []
    for n_shadow in n_shadow_models:
        process = Process(target=worker, args=(n_shadow, None))
        processes.append(process)
        process.start()

    for p in processes:
        p.join()