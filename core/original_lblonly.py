import math
import os
import sys

from bboxes.dtbb import DecisionTreeBlackBox
from core.attack_model import AttackModel

file_dir = os.path.dirname("..")
sys.path.append(file_dir)
from core.attack import Attack
from bboxes.nnbb import NeuralNetworkBlackBox
from bboxes.rfbb import RandomForestBlackBox
import pandas as pd
from math import ceil
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, accuracy_score, precision_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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


class Original_lblonly(Attack):
    def __init__(self, bb, NOISE_SAMPLES, db_name='adult', settings=[0, 0, 0], write_files=True):
        super().__init__(bb, database_name=db_name)
        self.NOISE_SAMPLES = NOISE_SAMPLES
        self.settings = settings
        self.scaler = None
        self.N_SHADOW_MODELS = 4
        self.model_type_folder = "regularized" if bb.regularized else "overfitted"
        self.model_name = bb.model_name
        self.write_files = write_files

    def robustness_score_label(self, model, dataset, label, n, scaler=None):
        """
        Compute the robustness score for each row inside the dataset with the true label passed
        as parameter and in case of miss classification we set the score to 0.
        :param model: model to get the labels.
        :param dataset:
        :param n: number of perturbations.
        :return: score of robustness is a value 0<rob_score<1
        """
        fb = 0.7  # probability of flipping one bit.
        con_vals = 6  # number of continues values before the bit ones.
        percentage_deviation = (0.1, 0.50)  # min max of the percentage of the value to add or subtrack.
        scores = []
        index = 0
        for row in tqdm(dataset):
            variations = []
            y_true = label[index]
            y_predicted = model.predict(np.array([row]))[0]
            if y_true == y_predicted:
                for i in range(n):
                    perturbed_row = row.copy()
                    if self.db_name == 'adult':
                        perturbed_row[:con_vals] = neighborhood_noise(perturbed_row[:con_vals], percentage_deviation)
                        perturbed_row[con_vals:] = bernoulli_noise(perturbed_row[con_vals:], fb)
                    elif self.db_name == 'bank' or self.db_name == 'synth':
                        perturbed_row = neighborhood_noise(perturbed_row, percentage_deviation)
                    variations.append(perturbed_row)
                variations = np.array(variations)
                output = model.predict(variations)
                score = np.mean(np.array(list(map(lambda x: 1 if x == y_true else 0, output))))
                scores.append(score)
            else:
                scores.append(0)
            index += 1
        return scores

    def carlini_binary_rand_robust(self, model, ds, ds_label, p, noise_samples=100, stddev=0.040, scaler=None):
        index = 0
        scores = []
        for row in tqdm(ds):
            label = ds_label[index]
            y = model.predict(np.array([row]))[0]
            if y == label:
                if self.db_name == 'adult':
                    noise = np.random.binomial(1, p, (noise_samples, len(row[6:])))
                    x_sampled = np.tile(np.copy(row), (noise_samples, 1))
                    x_noisy = np.invert(row[6:].astype(bool), out=np.copy(x_sampled[:, 6:]),
                                        where=noise.astype(bool)).astype(np.int32)
                    noise = stddev * np.random.randn(noise_samples, row[:6].shape[-1])
                    x_noisy = np.concatenate([x_sampled[:, :6] + noise, x_noisy], axis=1)
                    noise_values = model.predict(x_noisy)
                    score = np.mean(np.array(list(map(lambda x: 1 if x == label else 0, noise_values))))
                    scores.append(score)
                elif self.db_name == 'bank' or self.db_name == 'synth':
                    x_sampled = np.tile(np.copy(row), (noise_samples, 1))
                    noise = stddev * np.random.randn(noise_samples, row.shape[-1])
                    x_noisy = x_sampled + noise
                    noise_values = model.predict(x_noisy)
                    score = np.mean(np.array(list(map(lambda x: 1 if x == label else 0, noise_values))))
                    scores.append(score)
            else:  # Miss classification
                scores.append(0)
            index += 1

        return np.array(scores)

    def train_shadow_models(self):
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
            indexes = np.random.choice(tr.shape[0], ts.shape[0], replace=False)
            if self.settings[2] == 0:
                pred_tr_robustness = self.carlini_binary_rand_robust(shadow, tr[indexes], tr_l[indexes], noise_samples=self.NOISE_SAMPLES, p=0.6)
            else:
                pred_tr_robustness = self.robustness_score_label(shadow, tr[indexes], tr_l[indexes], self.NOISE_SAMPLES, scaler=self.scaler) # old implementation

            df_in = pd.DataFrame(pred_tr_robustness)
            df_in["class_label"] = pred_tr_labels[indexes]
            df_in["target_label"] = 1
            report = classification_report(tr_l[indexes], pred_tr_labels[indexes])
            print(report)
            # df_in.to_csv("scores_tr{}.csv".format(m))
            # _report = open("report_shadow_train{}.txt".format(m), "w")
            # write_report.write(report)

            # Test
            pred_labels = shadow.predict(ts)
            if self.settings[2] == 0:
                pred_ts_robustness = self.robustness_score_label(shadow, ts, ts_l, self.NOISE_SAMPLES, scaler=self.scaler) # old implementation
            else:
                pred_ts_robustness = self.carlini_binary_rand_robust(shadow, ts, ts_l, noise_samples=self.NOISE_SAMPLES, p=0.6)
            df_out = pd.DataFrame(pred_ts_robustness)
            df_out["class_label"] = pred_labels
            df_out["target_label"] = 0
            report = classification_report(ts_l, pred_labels)
            print(report)
            # write_report = open("nn_report_shadow_test{}.txt".format(m), "w")
            # write_report.write(report)
            # We merge the dataframes with IN/OUT target and we save it.
            df_final = pd.concat([df_in, df_out])
            # Save the dataset
            # df_final.to_csv("shadow_df{}.csv".format(m))
            self.attack_dataset.append(df_final)

    def train_shadow_model(self):
        """
        Here we train the shadow model on the noise_shadow.csv dataset, in order
        to imitate the black-box model.
        :return:
        """
        source_model = self.bb.train_model(self.noise_train_set.values, self.noise_train_label.values)
        pred_tr_labels = source_model.predict(self.noise_train_set.values)
        tr_report = classification_report(self.noise_train_label, pred_tr_labels)
        print("Train report")
        print(tr_report)
        print("Test report")
        pred_ts_labels = source_model.predict(self.noise_test_set.values)
        ts_report = classification_report(self.noise_test_label, pred_ts_labels)
        print(ts_report)
        self.shadow_model = source_model

    def attack_workflow(self):
        self.undersample_noise_training()
        if self.settings[0] == 0:
            self.train_shadow_model()
        else:
            self.train_shadow_models()

        self.perturb_datasets()
        self.train_test_attackmodel()

    def get_max_accuracy(self, y_true, probs, thresholds=None):
        """Return the max accuracy possible given the correct labels and guesses. Will try all thresholds unless passed.

        Args:
          y_true: True label of `in' or `out' (member or non-member, 1/0)
          probs: The scalar to threshold
          thresholds: In a blackbox setup with a shadow/source model, the threshold obtained by the source model can be passed
            here for attackin the target model. This threshold will then be used.

        Returns: max accuracy possible, accuracy at the threshold passed (if one was passed), the max precision possible,
         and the precision at the threshold passed.

        """
        if thresholds is None:
            fpr, tpr, thresholds = roc_curve(y_true, probs)

        accuracy_scores = []
        precision_scores = []
        reports = []
        for thresh in thresholds:
            thresholded = [1 if m > thresh else 0 for m in probs]
            accuracy_scores.append(accuracy_score(y_true, thresholded))
            precision_scores.append(precision_score(y_true, thresholded))
            report = classification_report(y_true, thresholded)
            reports.append(report)

        accuracies = np.array(accuracy_scores)
        precisions = np.array(precision_scores)
        max_accuracy = accuracies.max()
        max_precision = precisions.max()
        max_accuracy_threshold = thresholds[accuracies.argmax()]
        max_precision_threshold = thresholds[precisions.argmax()]
        report = reports[accuracies.argmax()]
        return max_accuracy, max_accuracy_threshold, max_precision, max_precision_threshold, report

    def undersample_noise_training(self):
        # We perform undersampling in order to have a balaned dataset for training.
        undersample = RandomUnderSampler(sampling_strategy="majority")
        self.noise_train_set, self.noise_train_label = undersample.fit_resample(self.noise_train_set,
                                                                                self.noise_train_label)
        self.noise_train_set, self.noise_train_label = shuffle(self.noise_train_set, self.noise_train_label)

    def perturb_datasets(self):
        # We merge the scores for the shadow perturbed data and we assign (1-0 in out label)
        if self.settings[0] == 0:
            self.noise_data_label = np.concatenate([np.ones(self.noise_test_set.shape[0]), np.zeros(self.noise_test_set.shape[0])], axis=0)
            indexes = np.random.choice(self.noise_train_set.shape[0], self.noise_test_set.shape[0], replace=False)
            if self.settings[2] == 0:
                print("Generating with paper noise")
                # Shadow data
                tr_scores = self.carlini_binary_rand_robust(self.shadow_model, self.noise_train_set.values[indexes], self.noise_train_label.values[indexes],
                                                            noise_samples=self.NOISE_SAMPLES, p=0.6, scaler=self.scaler)
                ts_scores = self.carlini_binary_rand_robust(self.shadow_model, self.noise_test_set.values, self.noise_test_label.values,
                                                            noise_samples=self.NOISE_SAMPLES, p=0.6, scaler=self.scaler)
            else:
                print("Generating with our noise")
                tr_scores = self.robustness_score_label(self.shadow_model, self.noise_train_set.values[indexes],
                                                            self.noise_train_label.values[indexes],
                                                            n=self.NOISE_SAMPLES)
                ts_scores = self.robustness_score_label(self.shadow_model, self.noise_test_set.values,
                                                            self.noise_test_label.values, n=self.NOISE_SAMPLES)

            # We merge the test and train balanced datasets.
            self.noise_data_scores = np.concatenate([tr_scores, ts_scores], axis=0)
        else:
            jointed_shadow_ds = pd.concat(self.attack_dataset)
            undersample = RandomUnderSampler(sampling_strategy="majority")
            self.noise_data_scores, self.noise_data_label = undersample.fit_resample(np.array(jointed_shadow_ds[0]).reshape(-1, 1),
                                                                                    jointed_shadow_ds['target_label'])


        # We merge the scores for the blackbox perturbed data and we assign (1-0 in out label)
        self.bb_data_label = np.concatenate([np.ones(self.test_set.shape[0]), np.zeros(self.test_set.shape[0])],
                                            axis=0)
        indexes = np.random.choice(self.train_set.shape[0], self.test_set.shape[0], replace=False)
        if self.settings[2] == 0:
            # Blackbox data
            target_tr_scores = self.carlini_binary_rand_robust(self.bb, self.train_set.values[indexes],
                                                               self.train_label.values[indexes],
                                                               noise_samples=self.NOISE_SAMPLES, p=0.6,
                                                               scaler=self.scaler)
            target_ts_scores = self.carlini_binary_rand_robust(self.bb, self.test_set.values, self.test_label.values,
                                                               noise_samples=self.NOISE_SAMPLES, p=0.6, scaler=self.scaler)
        else:
            # Blackbox data
            target_tr_scores = self.robustness_score_label(self.bb, self.train_set.values[indexes], self.train_label.values[indexes],
                                                               n=self.NOISE_SAMPLES)
            target_ts_scores = self.robustness_score_label(self.bb, self.test_set.values, self.test_label.values,
                                                           n=self.NOISE_SAMPLES)


        self.bb_data_scores = np.concatenate([target_tr_scores, target_ts_scores], axis=0)
        # Saving score ts
        tmp = pd.DataFrame(self.bb_data_scores, columns=['score'])
        tmp['taget'] = self.bb_data_label
        if self.write_files:
            tmp.to_csv("./test_score_dataset.csv", index=False)
        # Save score tr
        tmp2 = pd.DataFrame(self.noise_data_scores, columns=['score'])
        tmp2['taget'] = self.noise_data_label
        if self.write_files:
            tmp2.to_csv("./train_score_dataset.csv", index=False)


    def train_test_attackmodel(self):
        # Undersampling for 50-50 balanced test set.
        undersample = RandomUnderSampler(sampling_strategy="majority")
        self.bb_data_scores, self.bb_data_label = undersample.fit_resample(self.bb_data_scores.reshape(-1, 1), self.bb_data_label)
        self.bb_data_scores = np.ndarray.flatten(self.bb_data_scores)
        # get the test accuracy at the threshold selected on the source data
        if self.settings[1] == 1:
            mdl = AttackModel(np.array(self.noise_data_scores).reshape(-1, 1), self.noise_data_label, attack_type='perturb')
            # Training attack model
            pred = mdl.predict(np.array(self.noise_data_scores).reshape(-1, 1))
            report = classification_report(self.noise_data_label, pred)
            print(report)
            if self.write_files:
                f = open("../results/{}/{}/{}/originallblonly_attack_tr_{}.txt".format(self.db_name, self.model_name, self.model_type_folder, self.settings), "w")
                f.write(report)
                f.close()
            # Evaluation attack model
            pred = mdl.predict(np.array(self.bb_data_scores).reshape(-1, 1))
            report = classification_report(self.bb_data_label, pred)
            if self.write_files:
                self.save_roc_curve_data(self.bb_data_label, pred, "../results/{}/{}/{}/originallblonly_attack_roc_{}.csv".format(self.db_name, self.model_name, self.model_type_folder, self.settings))
            print("Final")
            print(report)
            if self.write_files:
                f = open("../results/{}/{}/{}/originallblonly_attack_ts_{}.txt".format(self.db_name,self.model_name, self.model_type_folder, self.settings), "w")
                f.write(report)
                f.close()
        else:
            acc_source, t, prec_source, tprec, report = self.get_max_accuracy(self.noise_data_label, self.noise_data_scores)
            print(report)
            if self.write_files:
                f = open("../results/{}/{}/{}/originallblonly_attack_tr_{}.txt".format(self.db_name, self.model_name, self.model_type_folder, self.settings), "w")
                f.write(report)
                f.write("\nThreshold choosed {}".format(t))
                f.close()
            acc_test_t, _, _, _, report = self.get_max_accuracy(self.bb_data_label, self.bb_data_scores, thresholds=[t])
            print(report)

            out = [1 if m > t else 0 for m in self.bb_data_scores]
            if self.write_files:
                self.save_roc_curve_data(self.bb_data_label, out, "../results/{}/{}/{}/originallblonly_attack_roc_{}.csv".format(self.db_name, self.model_name, self.model_type_folder, self.settings))

            print("Threshold choosed {}".format(t))
            if self.write_files:
                f = open("../results/{}/{}/{}/originallblonly_attack_ts_{}.txt".format(self.db_name, self.model_name, self.model_type_folder, self.settings), "w")
                f.write(report)
                f.write("\nThreshold choosed {}".format(t))
                f.close()


from multiprocessing import Pool


def worker_start(att_conf, NOISE_SAMPLES):
    ds_name = att_conf['ds_name']
    bb = RandomForestBlackBox(db_name=ds_name, regularized=att_conf['regularized'])
    att = Original_lblonly(bb, NOISE_SAMPLES, db_name=ds_name, settings=att_conf['setting'])
    att.start_attack()


if __name__ == "__main__":
    NOISE_SAMPLES = 1000
    ds_name = 'adult'
    regularized = True
    setting = [0, 0, 0]
    # bb = NeuralNetworkBlackBox(db_name=ds_name, regularized=regularized)
    bb = DecisionTreeBlackBox(db_name=ds_name, regularized=regularized, explainer=True)
    # bb = RandomForestBlackBox(db_name=ds_name, regularized=regularized)
    att = Original_lblonly(bb, NOISE_SAMPLES, db_name=ds_name, settings=setting, write_files=False)
    att.start_attack()
    """ds_names = ['adult', 'bank', 'synth']
    settings = [[0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0]]
    list_of_attacks = []
    for ds_name in ds_names:
        for sett in settings:
            list_of_attacks.append({'ds_name': ds_name, 'setting': sett, 'regularized': True})
    pool = Pool(processes=12)
    for att in list_of_attacks:
        pool.apply_async(worker_start, args=(att, NOISE_SAMPLES))
    # close the process pool
    pool.close()
    # wait for all tasks to finish
    pool.join()"""
