import pickle
import pandas as pd
import os
from imblearn.under_sampling import RandomUnderSampler
from numpy import savetxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from bboxes.rfbb import RandomForestBlackBox
from tqdm import tqdm
import numpy as np
import warnings
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from math import ceil

from my_label_only.robustness_score import robustness_score, robustness_score_label

warnings.filterwarnings("ignore")


def trainNNClassifier(input_dim, tr, tr_l):
    """
    Given the train set and train label, we train a RandomForestClassifier
    with the same behaviour as the blackbox.
    :param x:
    :param y:
    :return:
    """
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="tanh")(inputs)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, activation="tanh")(x)
    # x = layers.Dropout(0.1)(x)
    output = layers.Dense(2, activation="softmax")(x)
    opt = tf.optimizers.Adam(learning_rate=0.001)
    model = keras.Model(inputs=inputs, outputs=output, name="nn_bb_model")
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(tr, tr_l, epochs=7, batch_size=16)
    return model


def normalize(ds, scaler=None, dataframe=True):
    """
    Normalize the dataset in order to be fitted inside the model.
    :param ds: dataframe with the data to be scaled
    :param scaler: if you have already fitted the scaler you can pass it and reuse it.
    :return: scaled dataset
    """
    continuos_val = ds.values[:, :6] if dataframe else ds[:, :6]
    binary_vals = ds.values[:, 6:] if dataframe else ds[:, 6:]
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(continuos_val)
    normalized_arr = scaler.transform(continuos_val)
    return np.concatenate([normalized_arr, binary_vals], axis=1), scaler


# Load black box
bb = keras.models.load_model("models/nn/nn_blackbox2.h5")

TS_PERC = 0.2
N_SHADOW_MODELS = 8  # number of shadow models
noise_dataset = pd.read_csv("data/adult_noise_shadow_labelled")
noise_dataset_label = noise_dataset.pop("class")
no_tr, no_ts, no_tr_label, no_ts_label = train_test_split(noise_dataset, noise_dataset_label, stratify=noise_dataset_label,
                                                                test_size=TS_PERC, random_state=1)
# Here we normalize the training set and the test set
no_train_set, scaler = normalize(no_tr)
no_test_set, _ = normalize(no_ts, scaler)

tr_chunk_size = ceil(no_train_set.shape[0]/N_SHADOW_MODELS)  # chunk for the train set.
ts_chunk_size = ceil(no_test_set.shape[0]/N_SHADOW_MODELS)   # chunk for the test set.

for m in range(N_SHADOW_MODELS):
    # We take it's chunk of training data and test data
    tr = no_train_set[m*tr_chunk_size:(m*tr_chunk_size)+tr_chunk_size]
    tr_l = no_tr_label[m * tr_chunk_size:(m * tr_chunk_size) + tr_chunk_size]
    ts = no_test_set[m*ts_chunk_size:(m*ts_chunk_size)+ts_chunk_size]
    ts_l = no_ts_label[m*ts_chunk_size:(m*ts_chunk_size)+ts_chunk_size]

    # We perform undersampling
    undersample = RandomUnderSampler(sampling_strategy="majority")
    tr, tr_l = undersample.fit_resample(tr, tr_l)

    shadow = trainNNClassifier(tr.shape[1], tr, tr_l)

    # Report on training set
    pred_tr_proba = shadow.predict(tr, verbose=0)
    # pred_tr_robustness = robustness_score(shadow, tr, 100)  # old implementation
    pred_tr_robustness = robustness_score_label(shadow, tr, tr_l.values, 100)
    pred_tr = np.argmax(pred_tr_proba, axis=1)
    df_in = pd.DataFrame(pred_tr_robustness)
    df_in["class_label"] = pred_tr
    df_in["target_label"] = 1
    report = classification_report(tr_l, pred_tr)
    print(report)

    # Test
    pred_proba = shadow.predict(ts, verbose=0)
    pred_ts_labels = np.argmax(pred_proba, axis=1)
    # pred_ts_robustness = robustness_score(shadow, ts, 100)  # old implementation
    pred_ts_robustness = robustness_score_label(shadow, ts, ts_l.values, 100)
    df_out = pd.DataFrame(pred_ts_robustness)
    df_out["class_label"] = pred_ts_labels
    df_out["target_label"] = 0
    report = classification_report(ts_l, pred_ts_labels)
    print(report)

    # We merge the dataframes with IN/OUT target and we save it.
    df_final = pd.concat([df_in, df_out])
    df_final.to_csv("data/shadow_label_only_truelabel_nn/shadow_{}_predicted_ds".format(m), index=False)

path = "data/shadow_label_only_truelabel_nn"
attack_dataset = pd.concat([pd.read_csv(path+"/"+n) for n in os.listdir(path)])
classes = list(attack_dataset['class_label'].unique())
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

    # Prediction and report of the performancies.
    pred = mdl.predict(test_set.values)
    report = classification_report(test_label, pred)
    print(report)

    # Saving of the model.
    filename = 'attacker_nn/lblonly_attacker_truelabel_class_{}.sav'.format(c)
    pickle.dump(mdl, open(filename, 'wb'))

train_set = pd.read_csv("data/adult_original_train_set.csv")
test_set = pd.read_csv("data/adult_original_test_set.csv")
train_label = pd.read_csv("data/adult_original_train_label.csv")
test_label = pd.read_csv("data/adult_original_test_label.csv")
# Here we normalize the training set and the test set
train_set, scaler = normalize(train_set)
test_set, _ = normalize(test_set, scaler)

# Getting predict proba from the black box on tr and assign 1 as target_label
trainset_predict_proba = bb.predict(train_set, verbose=0)
class_labels = np.argmax(trainset_predict_proba, axis=1)
# trainset_predict_proba = robustness_score(bb, train_set, 100)
trainset_predict_proba = robustness_score_label(bb, train_set, train_label.values, 100)
df_in = pd.DataFrame(trainset_predict_proba)
df_in['target_label'] = 1
df_in['class_labels'] = class_labels

# Getting predict proba from the black box on ts and assign 0 as target_label
testset_predict_proba = bb.predict(test_set, verbose=0)
class_labels2 = np.argmax(testset_predict_proba, axis=1)
# testset_predict_proba = robustness_score(bb, test_set, 100)
testset_predict_proba = robustness_score_label(bb, test_set, test_label.values, 100)
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

for c in classes:
    print("Results for class: {}".format(c))
    # Obtain the correct attack model for the class c.
    att_c = pickle.load(open("attacker_nn/lblonly_attacker_truelabel_class_{}.sav".format(c), 'rb'))
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

write_report = open("attacker_nn/lblonly_truelabel_jointed_test_measures.txt", "w")
write_report.write(report)
