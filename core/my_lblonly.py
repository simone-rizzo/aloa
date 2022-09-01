import sys
from math import ceil
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from core.attack import Attack
import pandas as pd
from my_label_only.robustness_score import robustness_score


class My_lblonly(Attack):
    def __init__(self, N_SHADOW_MODELS, NOISE_SAMPLES):
        self.N_SHADOW_MODELS = N_SHADOW_MODELS
        self.NOISE_SAMPLES = NOISE_SAMPLES

    def attack_workflow(self):
        self.train_shadow_models()
        self.train_attack_models()
        self.test_attack()
        pass

    def trainRFClassifier(self, x, y):
        rf = RandomForestClassifier()
        rf.fit(x, y)
        return rf

    def train_attack_models(self):
        attack_dataset = pd.concat(self.attack_dataset) # Concatenate all the shadow datasets.
        self.attack_models = []
        classes = list(attack_dataset['class_label'].unique())
        for c in classes:
            print("Class:{}".format(c))
            tr = attack_dataset[attack_dataset['class_label'] == c] # filtering by class
            tr.pop('class_label') # we then remove the class label
            tr_label = tr.pop('target_label') # we want to predict he target label IN-OUT

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
            mdl = RandomForestClassifier()
            mdl.fit(train_set.values, train_label.values)

            # Prediction and report of the performances.
            pred = mdl.predict(test_set.values)
            report = classification_report(test_label, pred)
            print(report)
            write_report = open("report_attack_test{}.txt".format(c), "w")
            write_report.write(report)
            # We merge all the attack models
            self.attack_models.append(mdl)

    def train_shadow_models(self):
        tr_chunk_size = ceil(self.noise_train_set.shape[0] / N_SHADOW_MODELS)  # chunk for the train set.
        ts_chunk_size = ceil(self.noise_test_set.shape[0] / N_SHADOW_MODELS)  # chunk for the test set.
        self.attack_dataset = []

        # For each shadow model
        for m in tqdm(range(N_SHADOW_MODELS)):
            # We take it's chunk of training data and test data
            tr = self.noise_train_set.values[m * tr_chunk_size:(m * tr_chunk_size) + tr_chunk_size]
            tr_l = self.noise_train_label.values[m * tr_chunk_size:(m * tr_chunk_size) + tr_chunk_size]
            ts = self.noise_test_set.values[m * ts_chunk_size:(m * ts_chunk_size) + ts_chunk_size]
            ts_l = self.noise_test_label.values[m * ts_chunk_size:(m * ts_chunk_size) + ts_chunk_size]

            # We perform undersampling
            undersample = RandomUnderSampler(sampling_strategy="majority")
            tr, tr_l = undersample.fit_resample(tr, tr_l)

            # we train the model.
            shadow = self.trainRFClassifier(tr, tr_l)

            # Report on training set
            pred_tr_labels = shadow.predict(tr)
            pred_tr_robustness = robustness_score(shadow, tr, self.NOISE_SAMPLES) # old implementation
            df_in = pd.DataFrame(pred_tr_robustness)
            df_in["class_label"] = pred_tr_labels
            df_in["target_label"] = 1
            report = classification_report(tr_l, pred_tr_labels)
            print(report)
            write_report = open("report_shadow_train{}.txt".format(m), "w")
            write_report.write(report)

            # Test
            pred_labels = shadow.predict(ts)
            # pred_ts_robustness = robustness_score(shadow, ts, self.NOISE_SAMPLES) # old implementation
            pred_ts_robustness = robustness_score_label(shadow, ts, ts_l, self.NOISE_SAMPLES) # old implementation
            df_out = pd.DataFrame(pred_ts_robustness)
            df_out["class_label"] = pred_labels
            df_out["target_label"] = 0
            report = classification_report(ts_l, pred_labels)
            print(report)
            write_report = open("report_shadow_test{}.txt".format(m), "w")
            write_report.write(report)
            # We merge the dataframes with IN/OUT target and we save it.
            df_final = pd.concat([df_in, df_out])
            # Save the dataset
            df_final.to_csv("shadow_df{}.csv".format(m))
            self.attack_dataset.append(df_final)

    def test_attack(self):
        # Getting predict proba from the black box on tr and assign 1 as target_label
        trainset_predict_proba = robustness_score(self.bb, self.train_set.values, self.NOISE_SAMPLES) # old one
        class_labels = self.bb.predict(self.train_set.values)
        df_in = pd.DataFrame(trainset_predict_proba)
        df_in['target_label'] = 1
        df_in['class_labels'] = class_labels

        # Getting predict proba from the black box on ts and assign 0 as target_label
        testset_predict_proba = robustness_score(self.bb, self.test_set.values, self.NOISE_SAMPLES) #old one
        class_labels2 = self.bb.predict(self.test_set.values)
        df_out = pd.DataFrame(testset_predict_proba)
        df_out['target_label'] = 0
        df_out['class_labels'] = class_labels2

        # Merge the results
        df_final = pd.concat([df_in, df_out])
        df_final.to_csv("test_dataset_perturbed.csv") # we save the merged dataset.
        classes = list(df_final['class_labels'].unique())
        print(df_final['target_label'].value_counts())
        print(df_final['class_labels'].value_counts())

        # Undersampling
        ts_l = df_final.pop("target_label")
        undersample = RandomUnderSampler(sampling_strategy="majority")
        df_new, ts_l = undersample.fit_resample(df_final, ts_l)
        df_final = pd.concat([df_new, ts_l], axis=1)
        print(df_final.shape)
        test_l = []
        predicted = []

        for c, i in enumerate(classes):
            print("Results for class: {}".format(c))
            att_c = self.attack_models[i]

            # Filter the dataset for data of the same class_label
            test = df_final[df_final['class_labels'] == c]
            test.pop("class_labels")

            # Obtaining the target
            test_label = test.pop("target_label")
            pred = att_c.predict(test.values)
            report = classification_report(test_label, pred)
            print(report)
            test_l.extend(test_label.values)
            predicted.extend(pred)

        print("Jointed:")
        report = classification_report(test_l, predicted)
        print(report)


if __name__ == "__main__":
    NOISE_SAMPLES = 1
    NOISE_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else NOISE_SAMPLES
    N_SHADOW_MODELS = 8
    att = My_lblonly(N_SHADOW_MODELS, NOISE_SAMPLES)
    att.start_attack()