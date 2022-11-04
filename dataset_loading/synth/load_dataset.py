import pandas as pd
import os
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
print(os.getcwd())
pd.options.mode.chained_assignment = None  # default='warn'
df = pd.read_csv("../../data/synth/Gaussian_dataset.csv")

# Unnamed: 0 rappresentano gli indici va rimosso.
df.drop(columns=['Unnamed: 0'], inplace=True)

# Remove duplicates entries
df.drop_duplicates(inplace=True)

# Remove null column rows
print(df.info())

# We take the labels
# Altamente sbilanciato
# Lo bilanciamo
print(df['target'].value_counts())
labels = df.pop('target')

# We split 80-20
train_set, test_set, train_label, test_label = train_test_split(df, labels, stratify=labels,
                                                                test_size=0.20, random_state=0)
test_set.to_csv('../../data/synth/noise_shadow_old.csv', index=False)


train_set, test_set, train_label, test_label = train_test_split(train_set, train_label, stratify=train_label,
                                                                test_size=0.95, random_state=0)
train_set.to_csv('../../data/synth/original_train_set.csv', index=False)
test_set[:train_set.shape[0]].to_csv('../../data/synth/original_test_set.csv', index=False)
train_label.to_csv('../../data/synth/original_train_label.csv', index=False)
test_label[:train_set.shape[0]].to_csv('../../data/synth/original_test_label.csv', index=False)

"""shadow = pd.read_csv("data/bank/bank_shadow.csv")
shadow.drop(columns=['Unnamed: 0'], inplace=True)
shadow.drop_duplicates(inplace=True)
print(shadow.info())
shadow = shadow[shadow['nbr_60_89_days_past_due_not_worse'].notnull()]
print(shadow.info())
# Remove Nan values from the columns
print(shadow.columns[shadow.isnull().any()].tolist())
shadow.dropna(inplace=True)
print(shadow.columns[shadow.isnull().any()].tolist())
# See if it's balanced or not.
print(shadow['target'].value_counts())
shadow_l = shadow.pop('target')
undersample = RandomUnderSampler(sampling_strategy="majority")
shadow, shadow_l = undersample.fit_resample(shadow, shadow_l.values)
print(shadow.shape)
shadow = pd.DataFrame(shadow)
shadow_l = pd.DataFrame(shadow_l)
shadow.to_csv("data/bank/noise_shadow_old.csv", index=False)"""

