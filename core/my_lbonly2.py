import math

from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from bboxes.rfbb import RandomForestBlackBox
from core.attack import Attack
from my_label_only.robustness_score import *


class My_lblonly2(Attack):
    def __init__(self, bb, N_SHADOW_MODELS, N_NOISE_EXAMPLES):
        super().__init__(bb)
        self.N_SHADOW_MODELS = N_SHADOW_MODELS
        self.N_NOISE_EXAMPLES = N_NOISE_EXAMPLES

    def attack_workflow(self):
        self.train_shadow_models()
        self.train_attack_model()
        self.test_attack()

    def train_shadow_models(self):
        tr_batch = math.ceil(self.train_set.shape[0] / self.N_SHADOW_MODELS)
        ts_batch = math.ceil(self.test_set.shape[0] / self.N_SHADOW_MODELS)
        self.attack_datasets = []
        for i in range(self.N_SHADOW_MODELS):
            tr = self.train_set.values[i*tr_batch:i*tr_batch+tr_batch]
            tr_l = self.train_label.values[i * tr_batch:i * tr_batch + tr_batch]
            ts = self.test_set.values[i * ts_batch:i * ts_batch + ts_batch]
            ts_l = self.test_label.values[i * ts_batch:i * ts_batch + ts_batch]

            # We perform undersampling
            undersample = RandomUnderSampler(sampling_strategy="majority")
            tr, tr_l = undersample.fit_resample(tr, tr_l)

            # Now we train the model similar to the blackbox
            shadow = self.bb.train_model(tr, tr_l)

            # Training
            tr_out = shadow.predict(tr)
            tr_scores = robustness_score_label(shadow, tr, tr_l, self.N_NOISE_EXAMPLES)
            tr_df = pd.DataFrame(tr_scores)
            tr_df['target'] = 1
            tr_df['class_label'] = tr_out
            print(classification_report(tr_l, tr_out))

            # Test
            ts_out = shadow.predict(ts)
            ts_scores = robustness_score_label(shadow, ts, ts_l, self.N_NOISE_EXAMPLES)
            ts_df = pd.DataFrame(ts_scores)
            ts_df['target'] = 0
            ts_df['class_label'] = ts_out
            print(classification_report(ts_l, ts_out))

            merged_df = pd.concat([tr_df, ts_df], axis=0)
            self.attack_datasets.append(merged_df)

    def train_attack_model(self):
        attack_dataset = pd.concat(self.attack_datasets, axis=0)
        classes = list(attack_dataset['class_label'].unique())
        self.attack_models = []
        for c in classes:
            print("Class:{}".format(c))
            tr = attack_dataset[attack_dataset['class_label'] == c]
            tr.pop('class_label')
            tr_l = tr.pop('target')

            # Undersampling and splitting
            undersample = RandomUnderSampler(sampling_strategy="majority")
            tr, tr_l = undersample.fit_resample(tr, tr_l)

            train_set, test_set, train_label, test_label = train_test_split(tr, tr_l, stratify=tr_l,
                                                                            test_size=0.20, random_state=1)
            # We train the attacker model.
            mdl = RandomForestClassifier()
            mdl.fit(train_set.values, train_label.values)

            # Prediction and report of the performances.
            report = self.eval_model(mdl, test_set, test_label)
            print(report)

            # We merge all the attack models
            self.attack_models.append(mdl)

    def test_attack(self):
        ds_l = np.concatenate([np.ones(self.train_set.shape[0]), np.zeros(self.test_set.shape[0])],
                       axis=0)
        ds = pd.concat([self.train_set, self.test_set], axis=0)
        class_label = self.bb.predict(ds.values)
        ds['class_label'] = class_label

        undersample = RandomUnderSampler(sampling_strategy="majority")
        ds, ds_l = undersample.fit_resample(ds, ds_l)

        class_label = ds.pop("class_label")
        ds_scores = robustness_score_label(self.bb, ds.values, class_label, self.N_NOISE_EXAMPLES)
        df_final = pd.DataFrame(ds_scores)
        df_final['target'] = ds_l
        df_final['class_label'] = class_label

        test_l = []
        predicted = []
        classes = list(df_final['class_label'].unique())
        for c, i in enumerate(classes):
            print("Results for class: {}".format(c))
            att_c = self.attack_models[i]

            # Filter the dataset for data of the same class_label
            test = df_final[df_final['class_label'] == c]
            test.pop("class_label")

            # Obtaining the target
            test_label = test.pop("target")
            pred = att_c.predict(test.values)
            report = classification_report(test_label, pred)
            print(report)
            test_l.extend(test_label.values)
            predicted.extend(pred)

        print("Jointed:")
        report = classification_report(test_l, predicted)
        print(report)


if __name__ == "__main__":
    bb = RandomForestBlackBox()
    att = My_lblonly2(bb, 8, 100)
    att.start_attack()