import os
import sys

from core.my_lblonly import neighborhood_noise, bernoulli_noise

file_dir = os.path.dirname("..")
sys.path.append(file_dir)
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve
import numpy as np
from tqdm import tqdm
from bboxes.nnbb import NeuralNetworkBlackBox
from bboxes.rfbb import RandomForestBlackBox
from core.attack import Attack
from sklearn.metrics import roc_curve, accuracy_score, precision_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Original_lblonly(Attack):
    def __init__(self, bb, NOISE_SAMPLES, is_nn=False, db_name='adult', settings=[0, 0, 0]):
        super().__init__(bb, is_nn, database_name=db_name)
        self.NOISE_SAMPLES = NOISE_SAMPLES
        self.settings = settings
        self.scaler = None

    def robustness_score_label(self, model, dataset, label, n, scaler=None):
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
            y_predicted = model.predict(np.array([row]))[0]
            if y_true == y_predicted:
                for i in range(n):
                    perturbed_row = row.copy()
                    if self.db_name == 'adult':
                        perturbed_row[:con_vals] = neighborhood_noise(perturbed_row[:con_vals], percentage_deviation)
                        perturbed_row[con_vals:] = bernoulli_noise(perturbed_row[con_vals:], fb)
                    elif self.db_name == 'bank':
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
                elif self.db_name == 'bank':
                    x_sampled = np.tile(np.copy(row), (noise_samples, 1))
                    noise = stddev * np.random.randn(noise_samples, row.shape[-1])
                    x_noisy = x_sampled + noise
                    noise_values = model.predict(x_noisy)
                    score = np.mean(np.array(list(map(lambda x: 1 if x == label else 0, noise_values))))
                    scores.append(score)
            else:  # Miss classification
                # print("miss classified")
                scores.append(0)
            index += 1

        return np.array(scores)

    def train_shadow_model(self):
        """
        Here we train the shadow model on the adult_noise_shadow_labelled dataset, in order
        to imitate the black-box model.
        :return:
        """
        source_model = self.bb.train_model(self.noise_train_set.values, self.noise_train_label.values, epochs=250)
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
        self.train_shadow_model()
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

    def perturb_datasets(self):
        # We merge the scores for the shadow perturbed data and we assign (1-0 in out label)
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

    def train_test_attackmodel(self):
        # Undersampling for 50-50 balanced test set.
        undersample = RandomUnderSampler(sampling_strategy="majority")
        self.bb_data_scores, self.bb_data_label = undersample.fit_resample(self.bb_data_scores.reshape(-1, 1), self.bb_data_label)
        self.bb_data_scores = np.ndarray.flatten(self.bb_data_scores)
        # get the test accuracy at the threshold selected on the source data
        if self.settings[1] == 1:
            mdl = RandomForestClassifier()
            mdl.fit(np.array(self.noise_data_scores).reshape(-1, 1), self.noise_data_label)
            # Training attack model
            pred = mdl.predict(np.array(self.noise_data_scores).reshape(-1, 1))
            report = classification_report(self.noise_data_label, pred)
            print(report)
            # Evaluation attack model
            pred = mdl.predict(np.array(self.bb_data_scores).reshape(-1, 1))
            report = classification_report(self.bb_data_label, pred)
            print("Final")
            print(report)
        else:
            acc_source, t, prec_source, tprec, report = self.get_max_accuracy(self.noise_data_label, self.noise_data_scores)
            print(report)
            acc_test_t, _, _, _, report = self.get_max_accuracy(self.bb_data_label, self.bb_data_scores, thresholds=[t])
            print(report)
            print("Threshold choosed {}".format(t))
            write_report = open("bank_original_lblonly_nn.txt".format(self.NOISE_SAMPLES), "w")
            write_report.write(report)


if __name__ == "__main__":
    NOISE_SAMPLES = 1000
    # bb = RandomForestBlackBox()
    ds_name = 'bank'
    bb = NeuralNetworkBlackBox(db_name=ds_name)
    settings = [0, 1, 0] # first is shadow model or not, second train model or not, tird perturbation algorithm.
    # NOISE_SAMPLES = int(sys.argv[1]) if len(sys.argv)> 1 else NOISE_SAMPLES
    att = Original_lblonly(bb, NOISE_SAMPLES, True, db_name=ds_name, settings=settings)
    att.start_attack()
