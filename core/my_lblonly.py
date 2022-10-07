import sys
import os
import sys
file_dir = os.path.dirname("..")
sys.path.append(file_dir)
from math import ceil
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from bboxes.rfbb import RandomForestBlackBox
from core.attack import Attack
import pandas as pd
from bboxes.nnbb import NeuralNetworkBlackBox
import math


def bernoulli_noise(bin_values, p):
    """

    :param bin_values: index of the column in which we have binary values.
    :param p: porbability of changing the value.
    :return:
    """
    for i in range(len(bin_values)):
        r = np.random.uniform(0, 1)
        if r <= p:
            bin_values[i] = math.fabs(bin_values[i]-1)
    return bin_values


def neighborhood_noise(values, pd):
    """

    :param values: continuous values to be perturbed
    :param pd: percentage deviation (min, max) is the percentage of the value to add or subtrack.
    :return:
    """
    for i in range(len(values)):
        r = np.random.uniform(pd[0], pd[1])
        r = round(r, 2)
        if np.random.randint(2, size=1)[0] == 1:  # 50% of probability to be added or subtracted
            r *= -1
        values[i] += round(values[i] * r, 3)
    return values


class My_lblonly(Attack):
    def __init__(self, bb, N_SHADOW_MODELS, NOISE_SAMPLES, is_nn):
        super().__init__(bb, is_nn)
        self.N_SHADOW_MODELS = N_SHADOW_MODELS
        self.NOISE_SAMPLES = NOISE_SAMPLES
        self.scaler = None

    def attack_workflow(self):
        """
        Questo era quello che c'era prima
        self.train_shadow_models()
        self.train_attack_models()
        self.test_attack()
        Sotto di questo è tutto un debbugging. va rimosso
        """
        # Da qua in giù rimuovi.
        if self.is_nn:
            # Here we normalize the training set and the test set
            self.noise_train_set, self.scaler = self.normalize(self.noise_train_set, dataFrame=True)
            self.noise_test_set, _ = self.normalize(self.noise_test_set, self.scaler, dataFrame=True)
        # We perform undersampling
        # undersample = RandomUnderSampler(sampling_strategy="majority")
        # tr, tr_l = undersample.fit_resample(self.noise_train_set.values, self.noise_train_label.values)
        # we train the model.
        # model = self.bb.train_model(tr, tr_l, 100)
        testset_predict_proba = self.robustness_score_label(self.bb, self.test_set.values, self.train_label.values, self.NOISE_SAMPLES)
        print(testset_predict_proba)
        pass

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
            # write_report.write(report)
            # We merge all the attack models
            self.attack_models.append(mdl)
    
    def robustness_score_label(self, model, dataset: pd.DataFrame, label, n, scaler=None):
        """
        Compute the robustness score for each row inside the dataset with the true label passed
        as parameter and in case of miss classification we set the score to 0.
        :param model: model to get the labels.
        :param dataset:
        :param n: number of perturbations.
        :return: score of robustness is a value 0<rob_score<1
        """
        fb = 0.60  # probability of flipping one bit.
        con_vals = 6  # number of continues values before the bit ones.
        percentage_deviation = (0.1, 0.50)  # min max of the percentage of the value to add or subtrack.
        scores = []
        index = 0
        for row in tqdm(dataset):
            variations = []
            y_true = label[index]
            if self.is_nn and scaler:
                input_scaled, _ = self.normalize(np.array([row]), scaler, False)
                y_predicted = model.predict(input_scaled)
            else:
                y_predicted = model.predict(np.array([row]))[0]
            # y_predicted = np.argmax(y_predicted) if len(y_predicted) > 1 else y_predicted
            if y_true == y_predicted:
                for i in range(n):
                    perturbed_row = row.copy()
                    perturbed_row[:con_vals] = neighborhood_noise(perturbed_row[:con_vals], percentage_deviation)
                    perturbed_row[con_vals:] = bernoulli_noise(perturbed_row[con_vals:], fb)
                    variations.append(perturbed_row)
                variations = np.array(variations)
                if self.is_nn:
                    x_noisy, _ = self.normalize(variations, scaler, False)
                    output = model.predict(x_noisy)
                else:
                    output = model.predict(x_noisy)
                score = np.mean(np.array(list(map(lambda x: 1 if x == y_true else 0, output))))
                scores.append(score)
            else:
                scores.append(0)
            index += 1
        return scores

    def robustness_score(self, model, dataset: pd.DataFrame, n, scaler=None):
        """
        Compute the robustness score for each row inside the dataset.
        :param model: model to get the labels.
        :param dataset:
        :param n: number of perturbations.
        :return: score of robustness is a value 0<rob_score<1
        """
        fb = 0.60  # probability of flipping one bit.
        con_vals = 6  # number of continues values before the bit ones.
        percentage_deviation = (0.1, 0.80)  # min max of the percentage of the value to add or subtrack.
        scores = []
        index = 0
        for row in tqdm(dataset):
            variations = []
            for i in range(n):
                perturbed_row = row.copy()
                perturbed_row[:con_vals] = neighborhood_noise(perturbed_row[:con_vals], percentage_deviation)
                perturbed_row[con_vals:] = bernoulli_noise(perturbed_row[con_vals:], fb)
                variations.append(perturbed_row)
            variations = np.array(variations)
            if self.is_nn:
                x_noisy, _ = self.normalize(variations, scaler, False)
                output = model.predict(x_noisy)
            else:
                output = model.predict(variations)
            _, c = np.unique(output, return_counts=True)  # we obtain the count of the majority output.
            score = c.max()/n
            scores.append(score)
            index += 1
        return scores
        
    def train_shadow_models(self):
        if self.is_nn:
            # Here we normalize the training set and the test set
            self.noise_train_set, self.scaler = self.normalize(self.noise_train_set, dataFrame=True)
            self.noise_test_set, _ = self.normalize(self.noise_test_set, self.scaler, dataFrame=True)
        
        tr_chunk_size = ceil(self.noise_train_set.shape[0] / self.N_SHADOW_MODELS)  # chunk for the train set.
        ts_chunk_size = ceil(self.noise_test_set.shape[0] / self.N_SHADOW_MODELS)  # chunk for the test set.
        self.attack_dataset = []

        # For each shadow model
        for m in tqdm(range(self.N_SHADOW_MODELS)):
            # We take it's chunk of training data and test data
            tr = self.noise_train_set.values[m * tr_chunk_size:(m * tr_chunk_size) + tr_chunk_size]
            tr_l = self.noise_train_label.values[m * tr_chunk_size:(m * tr_chunk_size) + tr_chunk_size]
            ts = self.noise_test_set.values[m * ts_chunk_size:(m * ts_chunk_size) + ts_chunk_size]
            ts_l = self.noise_test_label.values[m * ts_chunk_size:(m * ts_chunk_size) + ts_chunk_size]

            # We perform undersampling
            undersample = RandomUnderSampler(sampling_strategy="majority")
            tr, tr_l = undersample.fit_resample(tr, tr_l)

            # we train the model.
            shadow = self.bb.train_model(tr, tr_l)

            # Report on training set
            pred_tr_labels = shadow.predict(tr)
            pred_tr_robustness = self.robustness_score(shadow, tr, self.NOISE_SAMPLES) # old implementation
            # pred_tr_robustness = self.robustness_score_label(shadow, tr,tr_l, self.NOISE_SAMPLES, scaler=self.scaler) # old implementation

            df_in = pd.DataFrame(pred_tr_robustness)
            df_in["class_label"] = pred_tr_labels
            df_in["target_label"] = 1
            report = classification_report(tr_l, pred_tr_labels)
            print(report)
            df_in.to_csv("scores_tr{}.csv".format(m))
            write_report = open("report_shadow_train{}.txt".format(m), "w")
            write_report.write(report)

            # Test
            pred_labels = shadow.predict(ts)
            pred_ts_robustness = self.robustness_score(shadow, ts, self.NOISE_SAMPLES) # old implementation
            # pred_ts_robustness = self.robustness_score_label(shadow, ts, ts_l, self.NOISE_SAMPLES, scaler=self.scaler) # old implementation
            df_out = pd.DataFrame(pred_ts_robustness)
            df_out["class_label"] = pred_labels
            df_out["target_label"] = 0
            report = classification_report(ts_l, pred_labels)
            print(report)
            write_report = open("nn_report_shadow_test{}.txt".format(m), "w")
            write_report.write(report)
            # We merge the dataframes with IN/OUT target and we save it.
            df_final = pd.concat([df_in, df_out])
            # Save the dataset
            # df_final.to_csv("shadow_df{}.csv".format(m))
            self.attack_dataset.append(df_final)

    def test_attack(self):
        if self.is_nn:
            # Here we normalize the training set and the test set
            self.train_set, self.scaler = self.normalize(self.train_set, dataFrame=True)
            self.test_set, _ = self.normalize(self.test_set, self.scaler, dataFrame=True)
        
        # Getting predict proba from the black box on tr and assign 1 as target_label
        trainset_predict_proba = self.robustness_score(self.bb, self.train_set.values, self.NOISE_SAMPLES) # old one
        # trainset_predict_proba = self.robustness_score_label(self.bb, self.train_set.values, self.train_label.values, self.NOISE_SAMPLES, scaler=self.scaler) # old one

        class_labels = self.bb.predict(self.train_set.values)
        df_in = pd.DataFrame(trainset_predict_proba)
        df_in['target_label'] = 1
        df_in['class_labels'] = class_labels

        # Getting predict proba from the black box on ts and assign 0 as target_label
        testset_predict_proba = self.robustness_score(self.bb, self.test_set.values, self.NOISE_SAMPLES) #old one
        # testset_predict_proba = self.robustness_score_label(self.bb, self.test_set.values, self.test_label.values, self.NOISE_SAMPLES) #old one
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
        write_report = open("mtlblonly_report_{}_{}.txt".format(self.N_SHADOW_MODELS, self.NOISE_SAMPLES), "w")
        write_report.write(report)


if __name__ == "__main__":
    NOISE_SAMPLES = 100
    N_SHADOW_MODELS = 16
    # bb = RandomForestBlackBox()
    bb = NeuralNetworkBlackBox()
    NOISE_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else NOISE_SAMPLES
    att = My_lblonly(bb, N_SHADOW_MODELS, NOISE_SAMPLES, is_nn=True)
    att.start_attack()