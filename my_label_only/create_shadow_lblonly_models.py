from math import ceil
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
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
train_set = pd.read_csv("../data/adult_noise_shadow_labelled")
train_label = train_set.pop("class")
train_set, test_set, train_label, test_label = train_test_split(train_set, train_label, stratify=train_label,
                                                                test_size=TS_PERC, random_state=1)
tr_chunk_size = ceil(train_set.shape[0] / N_SHADOW_MODELS)  # chunk for the train set.
ts_chunk_size = ceil(test_set.shape[0] / N_SHADOW_MODELS)  # chunk for the test set.

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
    pred_tr_robustness = robustness_score(shadow, tr, 100) # old implementation
    # pred_tr_robustness = robustness_score_label(shadow, tr, tr_l, 100)
    df_in = pd.DataFrame(pred_tr_robustness)
    df_in["class_label"] = pred_tr_labels
    df_in["target_label"] = 1
    report = classification_report(tr_l, pred_tr_labels)
    print(report)

    # Test
    pred_labels = shadow.predict(ts)
    pred_ts_robustness = robustness_score(shadow, ts, 100) # old implementation
    # pred_ts_robustness = robustness_score_label(shadow, ts, ts_l, 100)  # new implementation
    df_out = pd.DataFrame(pred_ts_robustness)
    df_out["class_label"] = pred_labels
    df_out["target_label"] = 0
    report = classification_report(ts_l, pred_labels)
    print(report)

    # We merge the dataframes with IN/OUT target and we save it.
    df_final = pd.concat([df_in, df_out])
    df_final.to_csv("../data/shadow_label_only/shadow_{}_predicted_ds".format(m), index=False)
    # df_final.to_csv("../data/shadow_label_only_truelabel/shadow_{}_predicted_ds_truelabel".format(m), index=False)