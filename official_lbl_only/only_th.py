import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, accuracy_score, precision_score, classification_report
from numpy import genfromtxt
import scikitplot as skplt
import matplotlib.pyplot as plt

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
  reports = []
  for thresh in thresholds:
    thresholded = [1 if m > thresh else 0 for m in probs]
    accuracy_scores.append(accuracy_score(y_true, thresholded))
    precision_scores.append(precision_score(y_true, thresholded))
    report = classification_report(y_true, thresholded)
    reports.append(report)

  accuracies = np.array(accuracy_scores)
  precisions = np.array(precision_scores)
  max_accuracy = accuracies.max()
  max_precision = precisions.max()
  max_accuracy_threshold = thresholds[accuracies.argmax()]
  max_precision_threshold = thresholds[precisions.argmax()]
  report = reports[accuracies.argmax()]
  return max_accuracy, max_accuracy_threshold, max_precision, max_precision_threshold, report


tr_score = genfromtxt('tr_scores.csv', delimiter=',')
ts_score = genfromtxt('ts_scores.csv', delimiter=',')
target_tr_scores = genfromtxt('target_tr_scores.csv', delimiter=',')
target_ts_scores = genfromtxt('target_ts_scores.csv', delimiter=',')

# True label for attack model (1-0 in out)
source_m = np.concatenate([np.ones(tr_score.shape[0]), np.zeros(ts_score.shape[0])], axis=0)
jointed_scores = np.concatenate([tr_score, ts_score], axis=0)

# True label for attack model (1-0 in out)
target_m = np.concatenate([np.ones(target_tr_scores.shape[0]), np.zeros(target_ts_scores.shape[0])], axis=0)
target_jointed_scores = np.concatenate([target_tr_scores, target_ts_scores], axis=0)

# get the test accuracy at the threshold selected on the source data
acc_source, t, prec_source, tprec, report = get_max_accuracy(source_m, jointed_scores)
print(report)
acc_test, _, prec_test, _, report = get_max_accuracy(target_m, target_jointed_scores)
print(report)
acc_test_t, _, _, _, report = get_max_accuracy(target_m, target_jointed_scores, thresholds=[t])
print(report)
_, _, prec_test_t, _, report = get_max_accuracy(target_m, target_jointed_scores, thresholds=[tprec])
# write_report = open("measures_lbl_only_original_attack_nn.txt", "w")
write_report = open("measures_lbl_only_original_attack.txt", "w")
write_report.write(report)
print("acc src: {}, acc test (best thresh): {}, acc test (src thresh): {}, thresh: {}".format(acc_source, acc_test,
                                                                                                acc_test_t, t))
print("prec src: {}, prec test (best thresh): {}, prec test (src thresh): {}, thresh: {}".format(prec_source, prec_test,
                                                                                               prec_test_t, tprec))
