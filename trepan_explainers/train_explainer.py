from sklearn import tree
from sklearn.metrics import classification_report
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

db_name = 'adult'
train_set = pd.read_csv("./trepan_dt_data.csv")
train_label = pd.read_csv("./trepan_dt_labels.csv")

train_set, test_set, train_label, test_label = train_test_split(train_set, train_label, stratify=train_label,
                                                               test_size=0.20, random_state=0)

# Trying to set less depth in order to have a better explaination.
best_param = {'criterion': 'entropy', 'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 5}

# best_param = {'criterion': 'entropy', 'max_depth': 400, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 5}
dt = tree.DecisionTreeClassifier(**best_param)

# dt = tree.DecisionTreeClassifier(max_depth=None, min_samples_leaf=10, min_samples_split=20)
dt.fit(train_set.values, train_label.values)
predictions1 = dt.predict(train_set.values)
out = dt.predict_proba(train_set.values)

print(dt.predict_proba(test_set.values))
report = classification_report(train_label, predictions1)

name = "less_depth"
print(report)
# write_report = open("train_nn_blackbox_regularized.txt", "w")
write_report = open("train_nn_{}.txt".format(name), "w")
write_report.write(report)

predictions = dt.predict(test_set)
report = classification_report(test_label, predictions)
print(report)
# write_report = open("test_nn_blackbox_regularized.txt", "w")
write_report = open("test_nn_{}.txt".format(name), "w")
write_report.write(report)

# filename = 'dt_blackbox_regularized.sav'
filename = 'nn_{}.sav'.format(name)
pickle.dump(dt, open(filename, 'wb'))
