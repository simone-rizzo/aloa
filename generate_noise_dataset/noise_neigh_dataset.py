import pandas as pd
import pickle
import numpy as np


perc = 9
percentage_deviation = (0.1, 0.25)  # min max of the percentage of the value to add or subtrack.
filename = "../data/adult_shadow.csv"
dataset_shadow = pd.read_csv(filename)
dataset_shadow.pop('Unnamed: 0')
dataset_shadow.pop("class")
n_col = dataset_shadow.shape[1]
n_rows = dataset_shadow.shape[0]
for c in range(n_col):
    percentage = int((perc/float(100))*n_rows)
    index_to_replace = np.random.choice([i for i in range(n_rows)], size=percentage)
    for ind in index_to_replace:
        sample = np.random.uniform(low=percentage_deviation[0], high=percentage_deviation[1], size=(1,))[0]
        sample = round(sample, 2)
        if dataset_shadow.iat[ind, c] == 0:
            dataset_shadow.iloc[ind, c] = sample
        else:
            if np.random.randint(2, size=1)[0] == 1:  # 50% of probability to be added or subtracted
                sample *= -1
            dataset_shadow.iloc[ind, c] += round(dataset_shadow.iat[ind, c]*sample, 3)

filename = "../data/noise_shadow.csv.csv"
dataset_shadow.to_csv(filename)
