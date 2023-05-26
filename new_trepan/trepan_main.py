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
from tqdm import tqdm
import os


def new_score(accuracy, depth, weight_accuracy=1.0, weight_depth=1.0):
    score = (weight_accuracy * accuracy) + (weight_depth * (1 / depth))
    return score


def new_score_sorting(config_list, gs_scores):
    scores = []
    for i, conf in enumerate(config_list):
        depth = conf['max_depth']
        sr = gs_scores[i]
        new_sr = new_score(sr, depth, weight_accuracy=0.2, weight_depth=0.1)
        scores.append((new_sr, sr, depth, conf))
    scores.sort(key=lambda x: x[0], reverse=True)
    """for s in scores:
        print("new_score:{} acc:{} depth:{}".format(round(s[0], 2), s[1], s[2]))"""
    return scores[0][3]


def train_explainer_for_plot(best_param, tr_set, tr_label, ts_set, ts_label, n_dept):
    """
    Only for Adult dataset and only for nn overfitted.
    :param best_param:
    :param tr_set:
    :param tr_label:
    :param ts_set:
    :param ts_label:
    :return:
    """
    dt = tree.DecisionTreeClassifier(**best_param)
    dt.fit(tr_set, tr_label)
    path_to_save = "./xai_tradeoff/{}".format(n_dept[0])
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    with open(path_to_save+"/best_param.txt", "w") as f:
        f.write(str(best_param))
    tr_fidelity = dt.predict(tr_set)
    report_tr_fidelity = classification_report(tr_label, tr_fidelity)
    # Save the tr fidelity
    with open(path_to_save+"/tr_fidelity.txt", "w") as f:
        f.write(report_tr_fidelity)
    ts_fidelity = dt.predict(ts_set)
    report_ts_fidelity = classification_report(ts_label, ts_fidelity)
    # Save the ts fidelity
    with open(path_to_save+"/ts_fidelity.txt", "w") as f:
        f.write(report_ts_fidelity)
    # Save the model
    pickle.dump(dt, open(path_to_save+"/model.sav", 'wb'))


def train_explainer(dataset_name, best_param, tr_set, tr_label, ts_set, ts_label, path_tuple, lss_dpt=False):
    dt = tree.DecisionTreeClassifier(**best_param)
    dt.fit(tr_set, tr_label)
    predictions1 = dt.predict(tr_set)
    report = classification_report(tr_label, predictions1)
    write_report = open("./explainers/{}/{}/{}/best_param{}.txt".format(dataset_name, path_tuple[0], path_tuple[1],
                                                                        "_lssdpt" if lss_dpt else ""), "w")
    write_report.write(str(best_param))
    write_report.close()

    write_report = open("./explainers/{}/{}/{}/tr_fidelity{}.txt".format(dataset_name, path_tuple[0], path_tuple[1],
                                                                         "_lssdpt" if lss_dpt else ""), "w")
    write_report.write(report)
    write_report.close()

    predictions = dt.predict(ts_set)
    report = classification_report(ts_label, predictions)

    write_report = open("./explainers/{}/{}/{}/ts_fidelity{}.txt".format(dataset_name, path_tuple[0], path_tuple[1],
                                                                         "_lssdpt" if lss_dpt else ""), "w")
    write_report.write(report)
    write_report.close()

    # Train fidelity
    fidelity_train = dt.predict(train_set)
    report = classification_report(train_label, fidelity_train)
    write_report = open("./explainers/{}/{}/{}/tr_original{}.txt".format(dataset_name, path_tuple[0], path_tuple[1],
                                                                         "_lssdpt" if lss_dpt else ""), "w")
    write_report.write(report)
    write_report.close()

    # Test fidelity
    fidelity_test = dt.predict(test_set)
    report = classification_report(test_label, fidelity_test)
    write_report = open(
        "./explainers/{}/{}/{}/ts_original{}.txt".format(dataset_name, path_tuple[0], path_tuple[1],
                                                         "_lssdpt" if lss_dpt else ""), "w")
    write_report.write(report)
    write_report.close()

    filename = "./explainers/{}/{}/{}/explainer{}.sav".format(dataset_name, path_tuple[0], path_tuple[1],
                                                              "_lssdpt" if lss_dpt else "")
    pickle.dump(dt, open(filename, 'wb'))
    write_report.close()


def greater_than_85(config_list, gs_scores):
    """
    We select the model having accuracy greater
    than 85 and the minimum number of depth.
    :param config_list:
    :param gs_scores:
    :return:
    """
    coupled = [(gs_scores[i], conf['max_depth'], conf) for i, conf in enumerate(config_list)]
    filtered = list(filter(lambda x: x[0] >= 0.77, coupled))
    filtered.sort(key=lambda x: x[0], reverse=False)
    print(filtered)
    return filtered[0][2]


def train_explainer_regularized(ds_name, train_set, train_label, test_set, test_label, path_tuple, max_depth_l=[8, 9, 10, 11, 12, 13]):
    tree_para = {'criterion': ['gini', 'entropy'], 'max_depth': max_depth_l,
                 'min_samples_split': [5, 10, 15, 25, 30, 50], 'min_samples_leaf': [3, 5, 15, 20, 40, 50],
                 'max_features': [5, 'auto', 'sqrt', 'log2']}
    grid = GridSearchCV(tree.DecisionTreeClassifier(), tree_para, cv=3, n_jobs=12, verbose=10, scoring='accuracy')
    grid.fit(train_set, train_label.ravel())
    configurations_list = grid.cv_results_['params']
    scores_list = grid.cv_results_['mean_test_score']
    # Select the way you want to perform the model selection
    best_param = new_score_sorting(configurations_list, scores_list)
    # best_param = greater_than_85(configurations_list, scores_list)
    # best_param = grid.best_params_
    # normal training of the explainer
    train_explainer(ds_name, best_param, train_set, train_label, test_set, test_label, path_tuple, lss_dpt=True)
    # train_explainer_for_plot(best_param, train_set, train_label, test_set, test_label, max_depth_l)



ds_names = ['synth']
regularizeds = [False, True]
dt_best_hyperparams = [
    {'criterion': 'entropy', 'max_depth': 20, 'max_features': 5, 'min_samples_leaf': 40, 'min_samples_split': 50},
    {'criterion': 'gini', 'max_depth': 13, 'splitter': 'best', 'max_features': None, 'min_samples_leaf': 1,
     'min_samples_split': 2},
]
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
    for bb in tqdm(bboxes):
        filename = "explainer_" + bb.model_name + "_{}_".format("regularized" if bb.regularized else "overfitted")
        # Here we generate the new data according to Trepan
        generator = TrePanGenerator()
        gen = generator.generate(train_set.values, oracle=bb, size=50000)
        data_l = gen[:, -1]
        data = np.delete(gen, -1, axis=1)
        tr_set, ts_set, tr_label, ts_label = train_test_split(data, data_l, stratify=data_l,
                                                              test_size=0.20, random_state=0)

        # Best params
        train_explainer(ds_name, dt_best_hyperparams[1], tr_set, tr_label, ts_set, ts_label,
                        (bb.model_name, "regularized" if bb.regularized else "overfitted"))
        # Regolarized with less depth
        train_explainer_regularized(ds_name, tr_set, tr_label, ts_set, ts_label,
                                        (bb.model_name, "regularized" if bb.regularized else "overfitted"), max_depth_l=[i for i in range(5, 20)])
