import pickle
import sys
from math import ceil
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report

from bboxes.rfbb import RandomForestBlackBox
from my_label_only.robustness_score import *
from tqdm import tqdm


def trainDTClassifier(x, y):
    """
    Given the train set and train label, we train a DecisionTreeClassifier
    with the same behaviour as the blackbox.
    :param x:
    :param y:
    :return:
    """
    dt = tree.DecisionTreeClassifier()
    dt.fit(x, y)
    return dt


def trainRFClassifier(x, y):
    """
    Given the train set and train label, we train a RandomForestClassifier
    with the same behaviour as the blackbox.
    :param x:
    :param y:
    :return:
    """
    rf = RandomForestClassifier()
    rf.fit(x, y)
    return rf


TS_PERC = 0.2
N_SHADOW_MODELS = 8
NOISE_SAMPLES = 1
NOISE_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else NOISE_SAMPLES
train_set = pd.read_csv("../data/adult_noise_shadow_labelled")
train_label = train_set.pop("class")
train_set, test_set, train_label, test_label = train_test_split(train_set, train_label, stratify=train_label,
                                                                test_size=TS_PERC, random_state=1)
tr_chunk_size = ceil(train_set.shape[0] / N_SHADOW_MODELS)  # chunk for the train set.
ts_chunk_size = ceil(test_set.shape[0] / N_SHADOW_MODELS)  # chunk for the test set.

attack_dataset = None
# For each shadow model
for m in tqdm(range(N_SHADOW_MODELS)):
    # We take it's chunk of training data and test data
    tr = train_set.values[m * tr_chunk_size:(m * tr_chunk_size) + tr_chunk_size]
    tr_l = train_label.values[m * tr_chunk_size:(m * tr_chunk_size) + tr_chunk_size]
    ts = test_set.values[m * ts_chunk_size:(m * ts_chunk_size) + ts_chunk_size]
    ts_l = test_label.values[m * ts_chunk_size:(m * ts_chunk_size) + ts_chunk_size]

    # We perform undersampling
    undersample = RandomUnderSampler(sampling_strategy="majority")
    tr, tr_l = undersample.fit_resample(tr, tr_l)

    # we train the model.
    # shadow = trainDTClassifier(tr, tr_l)
    shadow = trainRFClassifier(tr, tr_l)

    # Report on training set
    pred_tr_labels = shadow.predict(tr)
    # pred_tr_robustness = robustness_score(shadow, tr, 100) # old implementation
    pred_tr_robustness = robustness_score_label(shadow, tr, tr_l, NOISE_SAMPLES)
    df_in = pd.DataFrame(pred_tr_robustness)
    df_in["class_label"] = pred_tr_labels
    df_in["target_label"] = 1
    report = classification_report(tr_l, pred_tr_labels)
    print(report)

    # Test
    pred_labels = shadow.predict(ts)
    # pred_ts_robustness = robustness_score(shadow, ts, 100) # old implementation
    pred_ts_robustness = robustness_score_label(shadow, ts, ts_l, NOISE_SAMPLES)  # new implementation
    df_out = pd.DataFrame(pred_ts_robustness)
    df_out["class_label"] = pred_labels
    df_out["target_label"] = 0
    report = classification_report(ts_l, pred_labels)
    print(report)

    # We merge the dataframes with IN/OUT target and we save it.
    df_final = pd.concat([df_in, df_out])
    if attack_dataset is None:
        attack_dataset = df_final.copy()
    else:
        attack_dataset = pd.concat([attack_dataset, df_final])

# Train attacker models
classes = list(attack_dataset['class_label'].unique())
attackers = []
for c in classes:
    print("Class:{}".format(c))
    tr = attack_dataset[attack_dataset['class_label'] == c]
    tr.pop('class_label')
    tr_label = tr.pop('target_label')

    # Print of the unbalanced dataset
    unique, counts = np.unique(tr_label, return_counts=True)
    print(np.asarray((unique, counts)).T)

    # Undersampling and splitting
    undersample = RandomUnderSampler(sampling_strategy="majority")
    tr, tr_label = undersample.fit_resample(tr, tr_label)

    # Print after the balancing.
    unique, counts = np.unique(tr_label, return_counts=True)
    print(np.asarray((unique, counts)).T)

    train_set, test_set, train_label, test_label = train_test_split(tr, tr_label, stratify=tr_label,
                                                                    test_size=0.20, random_state=1)

    # We train the attacker model.
    # mdl = tree.DecisionTreeClassifier()
    mdl = RandomForestClassifier()
    mdl.fit(train_set.values, train_label.values)

    # Prediction and report of the performances.
    pred = mdl.predict(test_set.values)
    report = classification_report(test_label, pred)
    print(report)

    # Saving of the model.
    attackers.append(mdl)

# TEST
bb = RandomForestBlackBox()
train_set = pd.read_csv("../data/adult_original_train_set.csv")
train_label = pd.read_csv("../data/adult_original_train_label.csv")
test_set = pd.read_csv("../data/adult_original_test_set.csv")
test_label = pd.read_csv("../data/adult_original_test_label.csv")

# Getting predict proba from the black box on tr and assign 1 as target_label
# trainset_predict_proba = robustness_score(bb, train_set.values, 100) # old one
trainset_predict_proba = robustness_score_label(bb, train_set.values, train_label.values, NOISE_SAMPLES)
class_labels = bb.predict(train_set.values)
df_in = pd.DataFrame(trainset_predict_proba)
df_in['target_label'] = 1
df_in['class_labels'] = class_labels

# Getting predict proba from the black box on ts and assign 0 as target_label
# testset_predict_proba = robustness_score(bb, test_set.values, 100) #old one
testset_predict_proba = robustness_score_label(bb, test_set.values, test_label.values, NOISE_SAMPLES)
class_labels2 = bb.predict(test_set.values)
df_out = pd.DataFrame(testset_predict_proba)
df_out['target_label'] = 0
df_out['class_labels'] = class_labels2

# Merge the results
df_final = pd.concat([df_in, df_out])
classes = list(df_final['class_labels'].unique())
print(df_final['target_label'].value_counts())
print(df_final['class_labels'].value_counts())

test_l = []
predicted = []

for c, i in enumerate(classes):
    print("Results for class: {}".format(c))
    att_c = attackers[i]

    # Filter the dataset for data of the same class_label
    test = df_final[df_final['class_labels'] == c]
    test.pop("class_labels")

    # Obtaining the target
    test_label = test.pop("target_label")
    pred = att_c.predict(test.values)
    report = classification_report(test_label, pred)
    print(report)
    test_l.extend(test_label.values)
    predicted.extend(pred)

print("Jointed:")
report = classification_report(test_l, predicted)
print(report)
