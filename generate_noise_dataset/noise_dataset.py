import pandas as pd
import pickle
import numpy as np

perc = 5
filename = "../data/adult/adult_shadow.csv"
dataset_shadow = pd.read_csv(filename)
dataset_shadow.pop("class")
n_col = dataset_shadow.shape[1]
n_rows = dataset_shadow.shape[0]
for c in range(n_col):
    percentage = int((perc/float(100))*n_rows)
    index_to_replace = np.random.choice([i for i in range(n_rows)], size=percentage)
    new_values = np.random.rand(percentage)
    for ind, val in zip(index_to_replace, new_values):
        dataset_shadow.iloc[ind, c] = round(val, 2)

filename = "../data/adult/noise_shadow.csv"
dataset_shadow.to_csv(filename, index=False)
"""f = open(filename, 'wb')
pickle.dump(dataset_shadow, f)
f.close()"""