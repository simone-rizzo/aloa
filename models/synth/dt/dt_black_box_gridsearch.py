from sklearn import tree
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import classification_report
import pickle


db_name = 'synth'
train_set = pd.read_csv("../../../data/{}/original_train_set.csv".format(db_name))
test_set = pd.read_csv("../../../data/{}/original_test_set.csv".format(db_name))
train_label = pd.read_csv("../../../data/{}/original_train_label.csv".format(db_name))
test_label = pd.read_csv("../../../data/{}/original_test_label.csv".format(db_name))

tree_para = {'criterion': ['gini', 'entropy'], 'max_depth': [30, 40, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46],
             'min_samples_split': [4, 5, 6, 7, 8, 9, 10, 15, 25, 30, 50], 'min_samples_leaf': [3, 4, 5, 15, 20, 40, 50],
             'max_features': [1, 3, 5, 'auto', 'sqrt', 'log2']}
grid = GridSearchCV(tree.DecisionTreeClassifier(), tree_para, cv=5, n_jobs=12, verbose=10, scoring='accuracy')
grid.fit(train_set, train_label.values.ravel())
y_pred_acc = grid.predict(test_set)
report = classification_report(test_label, y_pred_acc)
print(report)
# write_report = open("models/dt/measures_grid_search_dt.txt", "w")
# write_report.write(report)
print(grid.best_params_)

"""title = "./best_param_gridsearch.txt"
with open(title, 'wb') as fp:
    fp.write(str(grid.best_params_))"""