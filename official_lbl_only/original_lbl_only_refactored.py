import sys
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
warnings.filterwarnings("ignore")


def carlini_binary_rand_robust(model, ds, ds_label, p, noise_samples=100, stddev=0.025):
    index = 0
    scores = []
    for row in tqdm(ds):
        label = ds_label[index]
        y = model.predict([row])[0]
        if y == label:
            noise = np.random.binomial(1, p, (noise_samples, len(row[6:])))
            x_sampled = np.tile(np.copy(row), (noise_samples, 1))
            x_noisy = np.invert(row[6:].astype(bool), out=np.copy(x_sampled[:, 6:]), where=noise.astype(bool)).astype(np.int32)
            noise = stddev * np.random.randn(noise_samples, row[:6].shape[-1])
            x_noisy = np.concatenate([x_sampled[:, :6] + noise, x_noisy], axis=1)
            noise_values = model.predict(x_noisy)
            score = np.mean(np.array(list(map(lambda x: 1 if x == label else 0, noise_values))))
            scores.append(score)
        else: # Miss classification
            print("miss classified")
            scores.append(0)
        index += 1

    return np.array(scores)


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


def train_target_model():
    """
    Here we train the shadow model on the noise_shadow.csv dataset, in order
    to imitate the black-box model.
    :return:
    """
    source_model = trainRFClassifier(tr.values, tr_l.values)
    pred_tr_labels = source_model.predict(tr.values)
    tr_report = classification_report(tr_l, pred_tr_labels)
    print("Train report")
    print(tr_report)
    print("Test report")
    pred_ts_labels = source_model.predict(source_test_set.values)
    ts_report = classification_report(source_test_label, pred_ts_labels)
    print(ts_report)
    return target_model


def get_max_accuracy(y_true, probs, thresholds=None):
  """Return the max accuracy possible given the correct labels and guesses. Will try all thresholds unless passed.

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
NOISE_SAMPLES = 5
NOISE_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else NOISE_SAMPLES

# Target model is the blackbox model that we want to perform the attack.
target_model = RandomForestBlackBox()
target_train_set = pd.read_csv("../data/original_train_set.csv")
target_train_label = pd.read_csv("../data/original_train_label.csv")
target_test_set = pd.read_csv("../data/original_test_set.csv")
target_test_label = pd.read_csv("../data/original_test_label.csv")


# Source model is model trained with similar data of the target one with the same architecture.
source_train_set = pd.read_csv("../data/noise_shadow.csv")
source_train_label = source_train_set.pop("class")
source_train_set, source_test_set, source_train_label, source_test_label = train_test_split(source_train_set, source_train_label, stratify=source_train_label,
                                                                test_size=TS_PERC, random_state=1)

# We perform undersampling in order to have a balaned dataset for training.
undersample = RandomUnderSampler(sampling_strategy="majority")
tr, tr_l = undersample.fit_resample(source_train_set, source_train_label)

source_model = train_target_model()

# Source
tr_scores = carlini_binary_rand_robust(source_model, tr.values, tr_l, noise_samples=NOISE_SAMPLES, p=0.6)
ts_scores = carlini_binary_rand_robust(source_model, source_test_set.values, source_test_label.values, noise_samples=NOISE_SAMPLES, p=0.6)
savetxt('tr_scores_balanced{}.csv'.format(NOISE_SAMPLES), tr_scores, delimiter=',')
savetxt('ts_scores_balanced{}.csv'.format(NOISE_SAMPLES), ts_scores, delimiter=',')

# Target
target_tr_scores = carlini_binary_rand_robust(target_model, target_train_set.values, target_train_label.values, noise_samples=NOISE_SAMPLES, p=0.6)
target_ts_scores = carlini_binary_rand_robust(source_model, target_test_set.values, target_test_label.values, noise_samples=NOISE_SAMPLES, p=0.6)
savetxt('target_tr_scores_balanced{}.csv'.format(NOISE_SAMPLES), target_tr_scores, delimiter=',')
savetxt('target_ts_scores_balanced{}.csv'.format(NOISE_SAMPLES), target_ts_scores, delimiter=',')

# True label for attack model (1-0 in out)
print(tr_scores.shape)
print(ts_scores.shape)
source_m = np.concatenate([np.ones(ts_scores.shape[0]), np.zeros(ts_scores.shape[0])], axis=0)
jointed_scores = np.concatenate([tr_scores[:ts_scores.shape[0]], ts_scores], axis=0)

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



