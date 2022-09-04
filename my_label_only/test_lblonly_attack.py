import pickle
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from bboxes.rfbb import RandomForestBlackBox
import bboxes
from bboxes.dtbb import DecisionTreeBlackBox

# Loading black box and datasets.
# bb = DecisionTreeBlackBox()
from my_label_only.robustness_score import *

bb = RandomForestBlackBox()

train_set = pd.read_csv("../data/adult_original_train_set.csv")
train_label = pd.read_csv("../data/adult_original_train_label.csv")
test_set = pd.read_csv("../data/adult_original_test_set.csv")
test_label = pd.read_csv("../data/adult_original_test_label.csv")

# Getting predict proba from the black box on tr and assign 1 as target_label
trainset_predict_proba = robustness_score(bb, train_set.values, 100) # old one
# trainset_predict_proba = robustness_score_label(bb, train_set.values, train_label.values, 100)
class_labels = bb.predict(train_set.values)
df_in = pd.DataFrame(trainset_predict_proba)
df_in['target_label'] = 1
df_in['class_labels'] = class_labels

# Getting predict proba from the black box on ts and assign 0 as target_label
testset_predict_proba = robustness_score(bb, test_set.values, 100) #old one
# testset_predict_proba = robustness_score_label(bb, test_set.values, test_label.values, 100)
class_labels2 = bb.predict(test_set.values)
df_out = pd.DataFrame(testset_predict_proba)
df_out['target_label'] = 0
df_out['class_labels'] = class_labels2


# Merge the results
df_final = pd.concat([df_in, df_out])
classes = list(df_final['class_labels'].unique())
print(df_final['target_label'].value_counts())
print(df_final['class_labels'].value_counts())

# Undersampling
ts_l = df_final.pop("target_label")
undersample = RandomUnderSampler(sampling_strategy="majority")
df_new, ts_l = undersample.fit_resample(df_final, ts_l)
df_final = pd.concat([df_new, ts_l], axis=1)
print(df_final.shape)

test_l = []
predicted = []

for c in classes:
    print("Results for class: {}".format(c))
    # Obtain the correct attack model for the class c.
    att_c = pickle.load(open("../attacker/lblonly_attacker_class_{}.sav".format(c), 'rb'))
    # att_c = pickle.load(open("../attacker/lblonly_attacker_truelabel_class_{}.sav".format(c), 'rb'))

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

# write_report = open("../attacker/lblonly_truelabel_jointed_test_measures.txt.txt", "w")
# write_report.write(report)
