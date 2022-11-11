import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, accuracy_score, recall_score
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from core.attack_model import AttackModel


def predict_th_model(th, data, label):
    th_data = list(map(lambda x: 0 if x < th else 1, data))
    report = classification_report(label, th_data)
    return report


def th_model(data, label, metric='p'):
    thsld = np.linspace(0, 1)
    results = []
    for t in thsld:
        th_data = list(map(lambda x: 0 if x <= t else 1, data))
        if metric == 'p':
            p = precision_score(label, th_data)
        elif metric == 'a':
            p = accuracy_score(label, th_data)
        elif metric == 'r':
            p = recall_score(label, th_data)
        results.append(p)
        # print(classification_report(label, th_data))
    return thsld[np.argmax(results)]

tr = pd.read_csv("./train_score_dataset.csv")
ts = pd.read_csv("./test_score_dataset.csv")

"""fig, (ax1, ax2) = plt.subplots(1, 2)
tr[tr['taget'] == 0]['score'].plot.kde(ax=ax1, label="Test scores")
tr[tr['taget'] == 1]['score'].plot.kde(ax=ax1, label="Train scores")
boxplot = tr.boxplot(column=['score'], by='taget', ax=ax2)
plt.title("Training score analysis")
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
ts[ts['taget'] == 0]['score'].plot.kde(ax=ax1, label="Test scores")
ts[ts['taget'] == 1]['score'].plot.kde(ax=ax1, label="Train scores")
boxplot = ts.boxplot(column=['score'], by='taget', ax=ax2)
plt.title("Test score analysis")
plt.show()"""

tr_avg = (tr[tr['taget'] == 1]['score'].mean() + tr[tr['taget'] == 0]['score'].mean())/2
print("Train scores avg:{}".format(tr[tr['taget'] == 1]['score'].mean()))
print("Test scores avg:{}".format(tr[tr['taget'] == 0]['score'].mean()))
print("Value that separates TR and TS:{}".format(tr_avg))

print("Separating the two cluster")
report = predict_th_model(tr_avg, ts['score'].values, ts['taget'].values)
print(report)

print("Separating with my th_model precision")
th = th_model(tr['score'].values, tr['taget'].values, metric='p')
print("Threshold: {}".format(th))
report = predict_th_model(th, ts['score'].values, ts['taget'].values)
print(report)

print("Separating with my th_model recall")
th = th_model(tr['score'].values, tr['taget'].values, metric='r')
print("Threshold: {}".format(th))
report = predict_th_model(th, ts['score'].values, ts['taget'].values)
print(report)

print("Separating with my th_model accuracy")
th = th_model(tr['score'].values, tr['taget'].values, metric='a')
print("Threshold: {}".format(th))
report = predict_th_model(th, ts['score'].values, ts['taget'].values)
print(report)

print("NN Model")
mdl = AttackModel(tr['score'].values.reshape(-1, 1), tr['taget'].values, attack_type='perturb')
print("Training:")
predicted = mdl.predict(tr['score'].values.reshape(-1, 1))
report = classification_report(tr['taget'].values, predicted)
print(report)
print("Test:")
predicted = mdl.predict(ts['score'].values.reshape(-1, 1))
report = classification_report(ts['taget'].values, predicted)
print(report)

param_grid = {
                    'bootstrap': [True, False],
                    'max_depth': [100, 350, 500],
                    'max_features': [5, 'auto', 'sqrt'],
                    'min_samples_leaf': [10, 20, 50],
                    'min_samples_split': [5, 10, 50],
                    'n_estimators': [100, 350, 500],
                    'criterion': ['gini', 'entropy']
        }
best_values = {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 100, 'max_features': 'sqrt', 'min_samples_leaf': 20, 'min_samples_split': 10, 'n_estimators': 350}
print("RF Model")
"""
Grid search for random forest.
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=12, verbose=2)
grid_search.fit(tr['score'].values.reshape(-1, 1), tr['taget'].values)
print(grid_search.best_params_)
predicted = grid_search.predict(tr['score'].values.reshape(-1, 1))
report = classification_report(tr['taget'].values, predicted)
print("TR:")
print(report)
predicted = grid_search.predict(ts['score'].values.reshape(-1, 1))
report = classification_report(ts['taget'].values, predicted)
print(report)"""
rf = RandomForestClassifier(**best_values)
rf.fit(tr['score'].values.reshape(-1, 1), tr['taget'].values)
predicted = rf.predict(tr['score'].values.reshape(-1, 1))
report = classification_report(tr['taget'].values, predicted)
print("Training:")
print(report)
predicted = rf.predict(ts['score'].values.reshape(-1, 1))
report = classification_report(ts['taget'].values, predicted)
print("Test")
print(report)

#create ROC curve
fpr, tpr, _ = metrics.roc_curve(ts['taget'].values, predicted)
plt.plot(fpr, tpr)
plt.plot([0.10*i for i in range(11)], [0.10*i for i in range(11)], '--', c="red")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("ROC curve")
plt.grid()
plt.show()
