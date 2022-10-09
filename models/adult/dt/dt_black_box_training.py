from sklearn import tree
from sklearn.metrics import classification_report
import pandas as pd
import pickle

train_set = pd.read_csv("../../data/adult_original_train_set_s45.csv")
test_set = pd.read_csv("../../data/adult_original_test_set_s45.csv")
train_label = pd.read_csv("../../data/adult_original_train_label_s45.csv")
test_label = pd.read_csv("../../data/adult_original_test_label_s45.csv")


dt = tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=30, max_features='auto',
    min_samples_leaf=2, min_samples_split=3)

# dt = tree.DecisionTreeClassifier()
dt.fit(train_set.values, train_label.values)
predictions1 = dt.predict(train_set.values)

print(dt.predict_proba(test_set.values))
report = classification_report(train_label, predictions1)
print(report)
write_report = open("measures_dt_black_box_training.txt", "w")
write_report.write(report)

predictions = dt.predict(test_set)
report = classification_report(test_label, predictions)
print(report)
write_report = open("measures_dt_black_box_test.txt", "w")
write_report.write(report)

filename = 'dt_black_box_original.sav'
pickle.dump(dt, open(filename, 'wb'))
