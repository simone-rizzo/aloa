import pandas as pd
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
df = pd.read_csv("./data/adult_data.csv", skipinitialspace=True, usecols=columns)

# Duplicated drop.
df = df.drop_duplicates()

# Deleting missing values.
df.drop(df.index[df['workclass'] == '?'], inplace=True)
df.drop(df.index[df['occupation'] == '?'], inplace=True)
df.drop(df.index[df['native-country'] == '?'], inplace=True)

# Binarizzation for the feature class salary for predicting purpose.
df.rename(columns={'salary': 'class'}, inplace=True)  # renaming the salary column to class.
df['class'] = df['class'].apply(lambda x: 0 if x == "<=50K" else 1)
categorical_classes = df.select_dtypes(include=['object']).columns.tolist()

# Hot encoding of all the categorical attributes.
df = pd.get_dummies(df, columns=categorical_classes)

label_dt = df.pop('class')
train_set, shadow_set, train_label, shadow_label = train_test_split(df, label_dt, stratify=label_dt,
                                                                    test_size=0.40, random_state=1)

# Saving the shadow set and the original dataset.
shadow_set['class'] = shadow_label.values
train_set['class'] = train_label.values
# This set will be used later to train and evaluate the model.
train_set.to_csv('./data/adult_original.csv', index=False)
# Thia shadow set will be used by the shadow dataset generator.
shadow_set.to_csv('./data/adult_shadow.csv', index=False)

train_label = train_set.pop('class')
# Splittig the original dataset train set with the tipical hold out percentage 80-20.
train_set, test_set, train_label, test_label = train_test_split(train_set, train_label, stratify=train_label,
                                                                test_size=0.30, random_state=45)
train_set.to_csv('./data/adult_original_train_set_s45.csv', index=False)
test_set.to_csv('./data/adult_original_test_set_s45.csv', index=False)
train_label.to_csv('./data/adult_original_train_label_s45.csv', index=False)
test_label.to_csv('./data/adult_original_test_label_s45.csv', index=False)