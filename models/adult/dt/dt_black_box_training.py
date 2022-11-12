from sklearn import tree
from sklearn.metrics import classification_report
import pandas as pd
import pickle

db_name = 'adult'
train_set = pd.read_csv("../../../data/{}/original_train_set.csv".format(db_name))
test_set = pd.read_csv("../../../data/{}/original_test_set.csv".format(db_name))
train_label = pd.read_csv("../../../data/{}/original_train_label.csv".format(db_name))
test_label = pd.read_csv("../../../data/{}/original_test_label.csv".format(db_name))

best_param = {'criterion': 'entropy', 'max_depth': 80, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 30}
# dt = tree.DecisionTreeClassifier(**best_param)

dt = tree.DecisionTreeClassifier()
dt.fit(train_set.values, train_label.values)
predictions1 = dt.predict(train_set.values)

print(dt.predict_proba(test_set.values))
report = classification_report(train_label, predictions1)
print(report)
# write_report = open("train_nn_blackbox_regularized.txt", "w")
write_report = open("train_nn_blackbox.txt", "w")
write_report.write(report)

predictions = dt.predict(test_set)
report = classification_report(test_label, predictions)
print(report)
# write_report = open("test_nn_blackbox_regularized.txt", "w")
write_report = open("test_nn_blackbox.txt", "w")
write_report.write(report)

# filename = 'dt_blackbox_regularized.sav'
filename = 'dt_blackbox.sav'
pickle.dump(dt, open(filename, 'wb'))
