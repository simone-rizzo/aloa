from bboxes.dtbb import DecisionTreeBlackBox
from bboxes.nnbb import NeuralNetworkBlackBox
from bboxes.rfbb import RandomForestBlackBox
from core.original_lblonly import Original_lblonly

import time
def worker_start(attack):
    attack.start_attack()


if __name__ == "__main__":
    NOISE_SAMPLES = 1000
    ds_names = ['bank', 'synth']
    regularized_values = [False, True]
    settings = [[0, 0, 0], [1, 1, 1]]
    list_of_attacks = []
    for ds_name in ds_names:
        for regul in regularized_values:
            for sett in settings:
                dt_bb = DecisionTreeBlackBox(db_name=ds_name, regularized=regul)
                rf_bb = RandomForestBlackBox(db_name=ds_name, regularized=regul)
                list_of_attacks.append(Original_lblonly(dt_bb, NOISE_SAMPLES, db_name=ds_name, settings=sett))
                list_of_attacks.append(Original_lblonly(rf_bb, NOISE_SAMPLES, db_name=ds_name, settings=sett))
    print(len(list_of_attacks))
    from multiprocessing import Pool
    pool = Pool(processes=12)
    for att in list_of_attacks:
        pool.apply_async(worker_start, args=(att,))
    # close the process pool
    pool.close()
    # wait for all tasks to finish
    pool.join()



"""
if __name__ == "__main__":
NOISE_SAMPLES = 1000
# bb = RandomForestBlackBox()
models = []
ds_names = ['adult', 'bank', 'synth']
# first is shadow model or not, second train model or not, third perturbation algorithm.
config_settings = [[0, 0, 0], [1, 1, 1]]
threads = []
for sett in config_settings:
    for ds_name in ds_names:
        for regularized in [False, True]:
            p = Process(target=go_attack, args=(NOISE_SAMPLES, ds_name, sett,))
            p.start()
            threads.append(p)
for p in threads:
    p.join()"""