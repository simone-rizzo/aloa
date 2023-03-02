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


rob_score_reg = pd.read_csv("./test_score_dataset2.csv")
rob_score = pd.read_csv("./test_score_dataset_old.csv")

rob_score['robustness score'] = rob_score['taget'].apply(lambda x: 'IN' if x == 1 else 'OUT')
rob_score_reg['robustness score'] = rob_score_reg['taget'].apply(lambda x: 'IN' if x == 1 else 'OUT')

reg_gap = rob_score_reg[rob_score_reg['robustness score'] == 'IN'].mean()-rob_score_reg[rob_score_reg['robustness score'] == 'OUT'].mean()
fig, (ax1, ax2) = plt.subplots(1, 2)
"""rob_score_reg[rob_score_reg['target'] == 'IN']['score'].plot.kde(ax=ax1, label="IN")
rob_score_reg[rob_score_reg['target'] == 'OUT']['score'].plot.kde(ax=ax1, label="OUT")"""
boxplot = rob_score_reg.boxplot(column=['score'], by='robustness score', ax=ax1)
ax1.set_title("Regularized NN IN-OUT GAP = 0.03")
boxplot2 = rob_score.boxplot(column=['score'], by='robustness score', ax=ax2)
ax2.set_title("Overfitted NN IN-OUT GAP = 0.16")
fig.suptitle('Robustness score analysis', fontsize=16)
ax1.set_yticks(np.arange(0, 1, 0.05))
ax2.set_yticks(np.arange(0, 1, 0.05))
plt.show()
