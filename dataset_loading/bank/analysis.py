import pandas as pd
import os
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'
df = pd.read_csv("data/bank/bank_original.csv")

# Unnamed: 0 rappresentano gli indici va rimosso.
df.drop(columns=['Unnamed: 0'], inplace=True)

# Remove duplicates entries
df.drop_duplicates(inplace=True)

# Remove null column rows
print(df.info())
df = df[df['monthly_income'].notnull()]
print(df.info())

# We take the labels
# Altamente sbilanciato
# Lo bilanciamo
print(df['target'].value_counts())
labels = df.pop('target')
# Undersampling of the dataset
undersample = RandomUnderSampler(sampling_strategy="majority")
tr, tr_l = undersample.fit_resample(df, labels.values)
tr = pd.DataFrame(tr)
tr_l = pd.DataFrame(tr_l)
# We split 80-20
train_set, test_set, train_label, test_label = train_test_split(tr, tr_l, stratify=tr_l,
                                                                test_size=0.20, random_state=0)

shadow = pd.read_csv("data/bank/bank_shadow.csv")
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

