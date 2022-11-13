import os
import sys

from bboxes.dtbb import DecisionTreeBlackBox
from core.attack_model import AttackModel

file_dir = os.path.dirname("..")
sys.path.append(file_dir)
from math import ceil
from multiprocessing import Process
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score
from bboxes.nnbb import NeuralNetworkBlackBox

from bboxes.rfbb import RandomForestBlackBox
from core.attack import Attack
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
"""
Confidence Attack is the official MIA attack proposed on the paper: https://www.cs.cornell.edu/~shmat/shmat_oak17.pdf
which uses the convidence vector probability.
"""


class ConfidenceAttack(Attack):
    def __init__(self, bb, N_SHADOW_MODELS, db_name='adult', multy_attack=False):
        super().__init__(bb, database_name=db_name)
        self.N_SHADOW_MODELS = N_SHADOW_MODELS
        self.shadow_models = []
        self.multy_attack = multy_attack
        self.model_type_folder = "regularized" if bb.regularized else "overfitted"
        self.model_name = bb.model_name

    def train_shadow_models(self):
        self.tr_chunk_size = ceil(self.noise_train_set.shape[0] / self.N_SHADOW_MODELS)  # chunk for the train set.
        self.ts_chunk_size = ceil(self.noise_test_set.shape[0] / self.N_SHADOW_MODELS)  # chunk for the test set.
        self.attack_dataset = []

        # For each shadow model
        for m in range(self.N_SHADOW_MODELS):
            # We take it's chunk of training data and test data
            tr = self.noise_train_set.values[m * self.tr_chunk_size:(m * self.tr_chunk_size) + self.tr_chunk_size]
            tr_l = self.noise_train_label.values[m * self.tr_chunk_size:(m * self.tr_chunk_size) + self.tr_chunk_size]
            ts = self.noise_test_set.values[m * self.ts_chunk_size:(m * self.ts_chunk_size) + self.ts_chunk_size]
            ts_l = self.noise_test_label.values[m * self.ts_chunk_size:(m * self.ts_chunk_size) + self.ts_chunk_size]

            # We perform undersampling
            undersample = RandomUnderSampler(sampling_strategy="majority")
            tr, tr_l = undersample.fit_resample(tr, tr_l)

            # we train the model.
            shadow = self.bb.train_model(tr, np.array(tr_l))

            # Report on training set
            pred_tr_labels = shadow.predict(tr)
            pred_tr_proba = shadow.predict_proba(tr)
            df_in = pd.DataFrame(pred_tr_proba)
            df_in["class_label"] = pred_tr_labels
            df_in["target_label"] = 1
            report = classification_report(tr_l, pred_tr_labels)
            print(report)

            # Test
            pred_labels = shadow.predict(ts)
            pred_proba = shadow.predict_proba(ts)
            df_out = pd.DataFrame(pred_proba)
            df_out["class_label"] = pred_labels
            df_out["target_label"] = 0
            report = classification_report(ts_l, pred_labels)
            print(report)

            # We add the shadow model to the list of shadow models.
            self.shadow_models.append(shadow)

            # We merge the dataframes with IN/OUT target and we save it.
            df_final = pd.concat([df_in, df_out])
            self.attack_dataset.append(df_final)

    def attack_workflow(self):
        self.train_shadow_models()
        self.train_attack_models()
        self.test_attack()
        pass

    def train_attack_models(self):
        attack_dataset = pd.concat(self.attack_dataset)
        classes = list(attack_dataset['class_label'].unique())
        self.attack_models = []
        if not self.multy_attack:
            attack_dataset.pop('class_label')
            tr_label = attack_dataset.pop('target_label')
            # Let's undersample
            undersample = RandomUnderSampler(sampling_strategy="majority")
            tr, tr_label = undersample.fit_resample(attack_dataset, tr_label)

            train_set, test_set, train_label, test_label = train_test_split(tr, tr_label, stratify=tr_label,
                                                                            test_size=0.20, random_state=1)


            mdl = AttackModel(train_set.values, train_label.values)
            self.th = self.th_model(train_set.values, train_label.values)
            print(self.th)
            pred = mdl.predict(train_set.values)
            report = classification_report(train_label, pred)
            print(report)
            f = open("../results/{}/{}/{}/confidence_attack_tr.txt".format(self.db_name, self.model_name, self.model_type_folder), "w")
            f.write(report)
            f.close()

            # Prediction and report of the performances.
            pred = mdl.predict(test_set.values)
            # pred = np.argmax(pred, axis=1)
            report = classification_report(test_label, pred)
            print(report)
            """f = open("../results/{}/nn/confidence_attack_vl.txt".format(self.db_name), "a")
            f.write(report)
            f.close()"""

            self.attack_model = mdl
        else:
            for c in classes:
                print("Class:{}".format(c))
                tr = attack_dataset[attack_dataset['class_label'] == c]
                tr.pop('class_label')
                tr_label = tr.pop('target_label')

                # Print of the unbalanced dataset
                unique, counts = np.unique(tr_label, return_counts=True)
                print(np.asarray((unique, counts)).T)

                # Undersampling and splitting
                undersample = RandomUnderSampler(sampling_strategy="majority")
                tr, tr_label = undersample.fit_resample(tr, tr_label)

                # Print after the balancing.
                unique, counts = np.unique(tr_label, return_counts=True)
                print(np.asarray((unique, counts)).T)

                train_set, test_set, train_label, test_label = train_test_split(tr, tr_label, stratify=tr_label,
                                                                                test_size=0.20, random_state=1)

                # We train the attacker model.
                mdl = AttackModel(train_set.values, train_label.values)
                pred = mdl.predict(train_set.values)
                report = classification_report(train_label, pred)
                print(report)

                # Prediction and report of the performances.
                pred = mdl.predict(test_set.values)
                report = classification_report(test_label, pred)
                print(report)

                # We merge all the attack models
                self.attack_models.append(mdl)

    def test_attack(self):
        # Getting predict proba from the black box on tr and assign 1 as target_label
        trainset_predict_proba = self.bb.predict_proba(self.train_set.values)
        class_labels = np.argmax(trainset_predict_proba, axis=1)
        df_in = pd.DataFrame(trainset_predict_proba)
        df_in['target_label'] = 1
        df_in['class_labels'] = class_labels

        # Getting predict proba from the black box on ts and assign 0 as target_label
        testset_predict_proba = self.bb.predict_proba(self.test_set.values)
        class_labels2 = np.argmax(testset_predict_proba, axis=1)
        df_out = pd.DataFrame(testset_predict_proba)
        df_out['target_label'] = 0
        df_out['class_labels'] = class_labels2

        # Merge the lblonly_improvments
        df_final = pd.concat([df_in, df_out])
        classes = list(df_final['class_labels'].unique())
        print(df_final['target_label'].value_counts())
        print(df_final['class_labels'].value_counts())

        ts_l = df_final.pop("target_label")
        print(df_final.shape)
        undersample = RandomUnderSampler(sampling_strategy="majority")
        df_new, ts_l = undersample.fit_resample(df_final, ts_l)
        df_final = pd.concat([df_new, ts_l], axis=1)
        print(df_new.shape)
        test_l = []
        predicted = []

        if not self.multy_attack:
            att_c = self.attack_model
            df_new.pop("class_labels")
            out = att_c.predict(df_new.values)
            report = classification_report(ts_l, out)
            self.save_roc_curve_data(ts_l.values, out, "../results/{}/nn/{}.csv".format(self.db_name, "confidence_roc"))
            print("Result:")
            f = open("../results/{}/{}/{}/confidence_attack_ts.txt".format(self.db_name, self.model_name,
                                                                              self.model_type_folder),"w")
            f.write(report)
            f.close()
            print(report)
            """print("Th model")
            pred = list(map(lambda x: 0 if max(x) < self.th else 1, df_new.values))
            report = classification_report(ts_l, pred)
            print(report)"""
        else:
            for c, i in enumerate(classes):
                print("Results for class: {}".format(c))
                # Obtain the correct attack model for the class c.
                att_c = self.attack_models[i]
                # Filter the dataset for data of the same class_label
                test = df_final[df_final['class_labels'] == c]
                test.pop("class_labels")

                # Obtaining the target
                test_label = test.pop("target_label")
                pred = att_c.predict(test.values)
                # pred = list(map(lambda x: 0 if max(x) < att_c else 1, test.values))
                report = classification_report(test_label, pred)
                print(report)
                test_l.extend(test_label.values)
                predicted.extend(pred)
            print("Jointed:")
            report = classification_report(test_l, predicted)
            print(report)

    def th_model(self, data, label):
        thsld = np.linspace(0, 1)
        results = []
        for t in thsld:
            th_data = list(map(lambda x: 0 if max(x) <= t else 1, data))
            p = precision_score(label, th_data)
            results.append(p)
            # print(classification_report(label, th_data))
        return thsld[np.argmax(results)]

    def predict_th_model(self, th, data, label):
        th_data = list(map(lambda x: 0 if max(x) < th else 1, data))
        report = classification_report(label, th_data)
        return report


if __name__ == "__main__":
    N_SHADOW_MODELS = 2
    # bb = RandomForestBlackBox()
    ds_name = 'adult'
    regularized = False
    bb = NeuralNetworkBlackBox(db_name=ds_name, regularized=regularized)
    # bb = DecisionTreeBlackBox(db_name=ds_name, regularized=regularized)
    # bb = RandomForestBlackBox(db_name=ds_name, regularized=regularized)
    att = ConfidenceAttack(bb, N_SHADOW_MODELS, db_name=ds_name, multy_attack=False)
    att.start_attack()
