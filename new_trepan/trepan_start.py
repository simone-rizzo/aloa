import pandas as pd
from bboxes.nnbb import NeuralNetworkBlackBox
#{'criterion': 'entropy', 'max_depth': 400, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 5}
from new_trepan.trepan_generation import TrePanGenerator
import numpy as np

ds_name = 'adult'
blackbox = 'nn'

train_set = pd.read_csv("../data/{}/original_train_set.csv".format(ds_name))
test_set = pd.read_csv("../data/{}/original_test_set.csv".format(ds_name))
train_label = pd.read_csv("../data/{}/original_train_label.csv".format(ds_name))
test_label = pd.read_csv("../data/{}/original_test_label.csv".format(ds_name))

bb = model = NeuralNetworkBlackBox(db_name=ds_name, regularized=False)
generator = TrePanGenerator()
gen = generator.generate(train_set.values, oracle=bb, size=70000)

labels = gen[:, -1]
gen = np.delete(gen, -1, axis=1)

gen = pd.DataFrame(gen)
labels = pd.DataFrame(labels)

gen.to_csv('./trepan_dt_data.csv', index=False)
labels.to_csv('./trepan_dt_labels.csv', index=False)



