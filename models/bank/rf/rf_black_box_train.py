from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import classification_report
import pickle
import json

train_set = pd.read_csv("../../data/original_train_set.csv")
test_set = pd.read_csv("../../data/original_test_set.csv")
train_label = pd.read_csv("../../data/original_train_label.csv")
test_label = pd.read_csv("../../data/original_test_label.csv")

# {'bootstrap': False, 'criterion': 'gini', 'max_depth': 100, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 5, 'n_estimators': 350}
rf = RandomForestClassifier(bootstrap=False, class_weight='balanced',
                       criterion='gini', max_depth=100, max_features='auto',
                       max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=10,
                       min_samples_split=5, min_weight_fraction_leaf=0.0,
                       n_estimators=350, n_jobs=None, oob_score=False,
                       random_state=0, verbose=0, warm_start=False)
# rf = RandomForestClassifier()
rf.fit(train_set, train_label.values.ravel())
filename = 'measures_rf_black_box_original_regularized.sav'
pickle.dump(rf, open(filename, 'wb'))

predictions1 = rf.predict(train_set)
score = rf.score(train_set.values, train_label.values.ravel())
report = classification_report(train_label, predictions1)
print(report)
write_report = open("measures_rf_black_box_training_regularized.txt", "w")
write_report.write(report)

predictions = rf.predict(test_set)
score = rf.score(test_set, test_label)
report = classification_report(test_label, predictions)
print(report)
write_report = open("measures_rf_black_box_test_regularized.txt", "w")
write_report.write(report)


