import pickle

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import tree
from bboxes.dtbb import DecisionTreeBlackBox
from bboxes.nnbb import NeuralNetworkBlackBox
from bboxes.rfbb import RandomForestBlackBox
from sklearn.model_selection import GridSearchCV
from new_trepan.trepan_generation import TrePanGenerator
import numpy as np


def new_score(accuracy, depth, weight_accuracy=1.0, weight_depth=1.0):
    score = (weight_accuracy * accuracy) + (weight_depth * (1/depth))
    return score


def new_score_sorting(config_list, gs_scores):
    scores = []
    for i, conf in enumerate(config_list):
        depth = conf['max_depth']
        sr = gs_scores[i]
        new_sr = new_score(sr, depth, weight_accuracy=0.2, weight_depth=0.1)
        scores.append((new_sr, sr, depth, conf))
    scores.sort(key=lambda x: x[0], reverse=True)
    for s in scores:
        print("new_score:{} acc:{} depth:{}".format(round(s[0], 2), s[1], s[2]))
    return scores[0][3]


def train_explainer(best_param, train_set, train_label, test_set, test_label, model_filename):
    dt = tree.DecisionTreeClassifier(**best_param)
    dt.fit(train_set, train_label)
    predictions1 = dt.predict(train_set)
    report = classification_report(train_label, predictions1)

    write_report = open("./explainers/{}/{}.txt".format('adult', model_filename+"train"), "w")
    write_report.write(report)
    write_report.close()

    predictions = dt.predict(test_set)
    report = classification_report(test_label, predictions)

    write_report = open("./explainers/{}/{}.txt".format('adult', model_filename + "test"), "w")
    write_report.write(report)
    write_report.close()

    filename = './explainers/{}/{}.sav'.format('adult', model_filename)
    pickle.dump(dt, open(filename, 'wb'))
    write_report.close()


def train_explainer_regularized(train_set, train_label, test_set, test_label, model_filename):
    tree_para = {'criterion': ['gini', 'entropy'], 'max_depth': [5, 6, 7, 8, 9],
                 'min_samples_split': [5, 10, 15, 25, 30, 50], 'min_samples_leaf': [3, 5, 15, 20, 40, 50],
                 'max_features': [1, 3, 5, 'auto', 'sqrt', 'log2']}
    grid = GridSearchCV(tree.DecisionTreeClassifier(), tree_para, cv=3, n_jobs=12, verbose=10, scoring='accuracy')
    grid.fit(train_set, train_label.ravel())
    configurations_list = grid.cv_results_['params']
    scores_list = grid.cv_results_['mean_test_score']
    best_param = new_score_sorting(configurations_list, scores_list)
    train_explainer(best_param, train_set, train_label, test_set, test_label, model_filename)


ds_names = ['adult']
regularizeds = [True, False]
dt_best_hyperparams = [{'criterion': 'entropy', 'max_depth': 80, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 30}, {'criterion': 'gini', 'max_depth': 13, 'splitter': 'best', 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}]
# For each dataset we have
for i, ds_name in enumerate(ds_names):
    # We load the dataset data
    train_set = pd.read_csv("../data/{}/original_train_set.csv".format(ds_name))
    test_set = pd.read_csv("../data/{}/original_test_set.csv".format(ds_name))
    train_label = pd.read_csv("../data/{}/original_train_label.csv".format(ds_name))
    test_label = pd.read_csv("../data/{}/original_test_label.csv".format(ds_name))
    bboxes = []
    # Here we allocate the 6 different bboxes
    for regularized in regularizeds:
        dt = DecisionTreeBlackBox(db_name=ds_name, regularized=regularized)
        rf = RandomForestBlackBox(db_name=ds_name, regularized=regularized)
        nn = NeuralNetworkBlackBox(db_name=ds_name, regularized=regularized)
        bboxes.append(dt)
        bboxes.append(rf)
        bboxes.append(nn)
    for bb in bboxes:
        filename = "explainer_"+bb.model_name+"_{}_".format("regularized" if bb.regularized else "overfitted")
        # Here we generate the new data according to Trepan
        generator = TrePanGenerator()
        gen = generator.generate(train_set.values, oracle=bb, size=70000)
        data_l = gen[:, -1]
        data = np.delete(gen, -1, axis=1)

        # Train with the best hyperparams
        tr_set, ts_set, tr_label, ts_label = train_test_split(data, data_l, stratify=data_l,
                                                                        test_size=0.20, random_state=0)

        # train_explainer(dt_best_hyperparams[i], tr_set, tr_label, ts_set, ts_label, filename)
        train_explainer_regularized(tr_set, tr_label, ts_set, ts_label, filename)
        print("finished")





