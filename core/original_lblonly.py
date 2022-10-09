import os
import sys
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
    def __init__(self, bb, NOISE_SAMPLES, is_nn=False):
        super().__init__(bb, is_nn)
        self.NOISE_SAMPLES = NOISE_SAMPLES
        self.scaler = None

    def carlini_binary_rand_robust(self, model, ds, ds_label, p, noise_samples=100, stddev=0.030, scaler=None):
        index = 0
        scores = []
        for row in tqdm(ds):
            label = ds_label[index]
            y = model.predict(np.array([row]))[0]
            if y == label:
                noise = np.random.binomial(1, p, (noise_samples, len(row[6:])))
                x_sampled = np.tile(np.copy(row), (noise_samples, 1))
                x_noisy = np.invert(row[6:].astype(bool), out=np.copy(x_sampled[:, 6:]),
                                    where=noise.astype(bool)).astype(np.int32)
                noise = stddev * np.random.randn(noise_samples, row[:6].shape[-1])
                x_noisy = np.concatenate([x_sampled[:, :6] + noise, x_noisy], axis=1)
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
        source_model = self.bb.train_model(self.noise_train_set.values, self.noise_train_label.values, epochs=400)
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
        # Shadow data
        tr_scores = self.carlini_binary_rand_robust(self.shadow_model, self.noise_train_set.values[indexes], self.noise_train_label.values[indexes],
                                                    noise_samples=self.NOISE_SAMPLES, p=0.6, scaler=self.scaler)
        ts_scores = self.carlini_binary_rand_robust(self.shadow_model, self.noise_test_set.values, self.noise_test_label.values,
                                                    noise_samples=self.NOISE_SAMPLES, p=0.6, scaler=self.scaler)
        # We merge the test and train balanced datasets.
        self.noise_data_scores = np.concatenate([tr_scores, ts_scores], axis=0)


        # We merge the scores for the blackbox perturbed data and we assign (1-0 in out label)
        self.bb_data_label = np.concatenate([np.ones(self.test_set.shape[0]), np.zeros(self.test_set.shape[0])],
                                            axis=0)
        indexes = np.random.choice(self.train_set.shape[0], self.test_set.shape[0], replace=False)
        # Blackbox data
        target_tr_scores = self.carlini_binary_rand_robust(self.bb, self.train_set.values[indexes], self.train_label.values[indexes],
                                                           noise_samples=self.NOISE_SAMPLES, p=0.6, scaler=self.scaler)
        target_ts_scores = self.carlini_binary_rand_robust(self.bb, self.test_set.values, self.test_label.values,
                                                           noise_samples=self.NOISE_SAMPLES, p=0.6, scaler=self.scaler)


        self.bb_data_scores = np.concatenate([target_tr_scores, target_ts_scores], axis=0)

    def train_test_attackmodel(self):
        # Undersampling for 50-50 balanced test set.
        undersample = RandomUnderSampler(sampling_strategy="majority")
        self.bb_data_scores, self.bb_data_label = undersample.fit_resample(self.bb_data_scores.reshape(-1, 1), self.bb_data_label)
        self.bb_data_scores = np.ndarray.flatten(self.bb_data_scores)
        # get the test accuracy at the threshold selected on the source data
        acc_source, t, prec_source, tprec, report = self.get_max_accuracy(self.noise_data_label, self.noise_data_scores)
        print(report)
        acc_test_t, _, _, _, report = self.get_max_accuracy(self.bb_data_label, self.bb_data_scores, thresholds=[t])
        print(report)
        print("Threshold choosed {}".format(t))
        write_report = open("original_lblonly_nn.txt".format(self.NOISE_SAMPLES), "w")
        write_report.write(report)


if __name__ == "__main__":
    NOISE_SAMPLES = 1000
    # bb = RandomForestBlackBox()
    bb = NeuralNetworkBlackBox()
    # NOISE_SAMPLES = int(sys.argv[1]) if len(sys.argv)> 1 else NOISE_SAMPLES
    att = Original_lblonly(bb, NOISE_SAMPLES, True)
    att.start_attack()
