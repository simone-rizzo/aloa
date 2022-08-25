import pandas as pd
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


warnings.filterwarnings("ignore")


def carlini_binary_rand_robust(model, ds, ds_label, scaler, p, noise_samples=100, stddev=0.025):
    index = 0
    scores = []
    for row in tqdm(ds):
        label = ds_label[index]
        input = np.array([row])
        input_scaled, _ = normalize(input, scaler, False)
        y = model.predict(input_scaled, verbose=0)
        y = 1 if y > 0.5 else 0
        if y == label:
            noise = np.random.binomial(1, p, (noise_samples, len(row[6:])))
            x_sampled = np.tile(np.copy(row), (noise_samples, 1))
            x_noisy = np.invert(row[6:].astype(bool), out=np.copy(x_sampled[:, 6:]), where=noise.astype(bool)).astype(np.int32)
            noise = stddev * np.random.randn(noise_samples, row[:6].shape[-1])
            x_noisy = np.concatenate([x_sampled[:, :6] + noise, x_noisy], axis=1)
            x_noisy, _ = normalize(x_noisy, scaler, False)
            noise_values = model.predict(x_noisy, verbose=0)
            noise_values = np.array(list(map(lambda x: 1 if x > 0.5 else 0, noise_values)))
            score = np.mean(np.array(list(map(lambda x: 1 if x == label else 0, noise_values))))
            scores.append(score)
        else: # Miss classification
            scores.append(0)
        index += 1

    return np.array(scores)


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
    output = layers.Dense(1, activation="sigmoid")(x)
    opt = tf.optimizers.Adam(learning_rate=0.001)
    model = keras.Model(inputs=inputs, outputs=output, name="nn_bb_model")
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    model.fit(tr, tr_l, epochs=30, batch_size=16)
    return model


def train_target_model(tr, tr_l, ts, ts_l):
    """
    Here we train the shadow model on the adult_noise_shadow_labelled dataset, in order
    to imitate the black-box model.
    :return:
    """
    source_model = trainNNClassifier(tr.shape[1], tr, tr_l)
    pred_tr_labels = source_model.predict(tr, verbose = 0)
    pred_tr_labels = np.array(list(map(lambda x: 1 if x > 0.5 else 0, pred_tr_labels)))
    tr_report = classification_report(tr_l, pred_tr_labels)
    print("Train report")
    print(tr_report)
    print("Test report")
    pred_ts_labels = source_model.predict(ts, verbose = 0)
    pred_ts_labels = np.array(list(map(lambda x: 1 if x > 0.5 else 0, pred_ts_labels)))
    ts_report = classification_report(ts_l, pred_ts_labels)
    print(ts_report)
    return target_model


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


def get_max_accuracy(y_true, probs, thresholds=None):
  """
  Return the max accuracy possible given the correct labels and guesses. Will try all thresholds unless passed.

  Args:
    y_true: True label of `in' or `out' (member or non-member, 1/0)
    probs: The scalar to threshold
    thresholds: In a blackbox setup with a shadow/source model, the threshold obtained by the source model can be passed
      here for attackin the target model. This threshold will then be used.

  Returns: max accuracy possible, accuracy at the threshold passed (if one was passed), the max precision possible,
   and the precision at the threshold passed.

  """
  if thresholds is None:
    fpr, tpr, thresholds = roc_curve(y_true, probs)

  accuracy_scores = []
  precision_scores = []
  for thresh in thresholds:
    accuracy_scores.append(accuracy_score(y_true,
                                          [1 if m > thresh else 0 for m in probs]))
    precision_scores.append(precision_score(y_true, [1 if m > thresh else 0 for m in probs]))

  accuracies = np.array(accuracy_scores)
  precisions = np.array(precision_scores)
  max_accuracy = accuracies.max()
  max_precision = precisions.max()
  max_accuracy_threshold = thresholds[accuracies.argmax()]
  max_precision_threshold = thresholds[precisions.argmax()]
  return max_accuracy, max_accuracy_threshold, max_precision, max_precision_threshold


TS_PERC = 0.2
N_SHADOW_MODELS = 8
# Target model is the blackbox model that we want to perform the attack.
target_model = keras.models.load_model("../models/nn/nn_blackbox.h5")

target_train_set = pd.read_csv("../data/adult_original_train_set.csv")
target_train_label = pd.read_csv("../data/adult_original_train_label.csv")

target_test_set = pd.read_csv("../data/adult_original_test_set.csv")
target_test_label = pd.read_csv("../data/adult_original_test_label.csv")

target_train_set_scaled, scaler = normalize(target_train_set)
target_test_set_scaled, _ = normalize(target_test_set, scaler)


# Source model is model trained with similar data of the target one with the same architecture.
source_train_set = pd.read_csv("../data/adult_noise_shadow_labelled")
source_train_label = source_train_set.pop("class")
source_train_set, source_test_set, source_train_label, source_test_label = train_test_split(source_train_set, source_train_label, stratify=source_train_label,
                                                                test_size=TS_PERC, random_state=1)


source_test_set_scaled, _ = normalize(source_test_set, scaler)

# We perform undersampling in order to have a balaned dataset for training.
undersample = RandomUnderSampler(sampling_strategy="majority")
tr, tr_l = undersample.fit_resample(source_train_set, source_train_label)
tr_scaled, _ = normalize(tr, scaler)

source_model = train_target_model(tr_scaled, tr_l, source_test_set_scaled, source_test_label.values)

# Source
ts_scores = carlini_binary_rand_robust(source_model, source_test_set.values, source_test_label.values, scaler, p=0.5)
tr_scores = carlini_binary_rand_robust(source_model, tr.values, tr_l, scaler, p=0.5)
savetxt('tr_scores_nn.csv', tr_scores, delimiter=',')
savetxt('ts_scores_nn.csv', ts_scores, delimiter=',')

# Target
target_tr_scores = carlini_binary_rand_robust(target_model, target_train_set.values, target_train_label.values, scaler, p=0.5)
target_ts_scores = carlini_binary_rand_robust(source_model, target_test_set.values, target_test_label.values, scaler, p=0.5)
savetxt('target_tr_scores_nn.csv', target_tr_scores, delimiter=',')
savetxt('target_ts_scores_nn.csv', target_ts_scores, delimiter=',')

# True label for attack model (1-0 in out)
source_m = np.concatenate([np.ones(tr_scores.shape[0]), np.zeros(ts_scores.shape[0])], axis=0)
jointed_scores = np.concatenate([tr_scores, ts_scores], axis=0)

# True label for attack model (1-0 in out)
target_m = np.concatenate([np.ones(target_tr_scores.shape[0]), np.zeros(target_ts_scores.shape[0])], axis=0)
target_jointed_scores = np.concatenate([target_tr_scores, target_ts_scores], axis=0)


# get the test accuracy at the threshold selected on the source data
acc_source, t, prec_source, tprec = get_max_accuracy(source_m, jointed_scores)
acc_test, _, prec_test, _ = get_max_accuracy(target_m, target_jointed_scores)
acc_test_t, _, _, _ = get_max_accuracy(target_m, target_jointed_scores, thresholds=[t])
_, _, prec_test_t, _ = get_max_accuracy(target_m, target_jointed_scores, thresholds=[tprec])
print("acc src: {}, acc test (best thresh): {}, acc test (src thresh): {}, thresh: {}".format(acc_source, acc_test,
                                                                                                acc_test_t, t))
print("prec src: {}, prec test (best thresh): {}, prec test (src thresh): {}, thresh: {}".format(prec_source, prec_test,
                                                                   prec_test_t, tprec))



