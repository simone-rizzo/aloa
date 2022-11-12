from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import classification_report
import pickle
import json

db_name = 'synth'
train_set = pd.read_csv("../../../data/{}/original_train_set.csv".format(db_name))
test_set = pd.read_csv("../../../data/{}/original_test_set.csv".format(db_name))
train_label = pd.read_csv("../../../data/{}/original_train_label.csv".format(db_name))
test_label = pd.read_csv("../../../data/{}/original_test_label.csv".format(db_name))

best_param = {'bootstrap': True, 'criterion': 'gini', 'max_depth': 100, 'max_features': 5, 'min_samples_leaf': 10, 'min_samples_split': 5, 'n_estimators': 350}
rf = RandomForestClassifier(**best_param)
# Overfitted
# rf = RandomForestClassifier()
rf.fit(train_set, train_label.values.ravel())
# filename = 'rf_blackbox.sav'
filename = 'rf_blackbox_regularized.sav'
pickle.dump(rf, open(filename, 'wb'))

predictions1 = rf.predict(train_set)
score = rf.score(train_set.values, train_label.values.ravel())
report = classification_report(train_label, predictions1)
print(report)
# write_report = open("train_rf_blackbox.txt", "w")
write_report = open("train_rf_blackbox_regularized.txt", "w")
write_report.write(report)

predictions = rf.predict(test_set)
score = rf.score(test_set, test_label)
report = classification_report(test_label, predictions)
print(report)
# write_report = open("test_rf_blackbox.txt", "w")
write_report = open("test_rf_blackbox_regularized.txt", "w")
write_report.write(report)


