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

param_grid = {
                    'bootstrap': [True, False],
                    'max_depth': [100, 350, 500],
                    'max_features': [5, 'auto', 'sqrt'],
                    'min_samples_leaf': [10, 20, 50],
                    'min_samples_split': [5, 10, 50],
                    'n_estimators': [100, 350, 500],
                    'criterion': ['gini', 'entropy']
        }
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=12, verbose=2)
grid_search.fit(train_set.values, train_label.values.ravel())
print(grid_search.best_params_)
y_pred_acc = grid_search.predict(test_set)
report = classification_report(test_label, y_pred_acc)
# write_report = open("measures_grid_search_rf.txt", "w")
# write_report.write(report)
print(grid_search.best_params_)
"""title = "best_param_rf.txt"
with open(title, 'wb') as fp:
    fp.write(json.dump(grid_search.best_params_))"""
