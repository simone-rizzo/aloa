import abc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from bboxes.rfbb import RandomForestBlackBox


class Attack(metaclass=abc.ABCMeta):

    def __init__(self, bbmodel):
        self.bb = bbmodel

    @abc.abstractmethod
    def attack_workflow(self):
        pass

    def start_attack(self):
        self.initialize_dataset()
        self.split_noise_dataset()
        self.attack_workflow()

    def initialize_dataset(self):
        self.noise_train_set = pd.read_csv("data/adult_noise_shadow_labelled")
        self.noise_train_label = self.noise_train_set.pop("class")
        self.train_set = pd.read_csv("data/adult_original_train_set.csv")
        self.train_label = pd.read_csv("data/adult_original_train_label.csv")
        self.test_set = pd.read_csv("data/adult_original_test_set.csv")
        self.test_label = pd.read_csv("data/adult_original_test_label.csv")

    def split_noise_dataset(self):
        self.noise_train_set, self.noise_test_set, self.noise_train_label, self.noise_test_label = train_test_split(
            self.noise_train_set, self.noise_train_label, stratify=self.noise_train_label,
            test_size=0.2, random_state=1)

    def eval_model(self, model, x, y):
        out = model.predict(x)
        return classification_report(y, out)

    def normalize(self, ds, scaler=None, dataFrame=True):
        """
        Normalize the dataset in order to be fitted inside the model.
        :param ds: dataframe with the data to be scaled
        :param scaler: if you have already fitted the scaler you can pass it and reuse it.
        :return: scaled dataset
        """
        continuos_val = ds.values[:, :6] if dataFrame else ds[:, :6]
        binary_vals = ds.values[:, 6:] if dataFrame else ds[:, 6:]
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(continuos_val)
        normalized_arr = scaler.transform(continuos_val)
        if dataFrame:
            return pd.DataFrame(np.concatenate([normalized_arr, binary_vals], axis=1)), scaler
        else:
            np.concatenate([normalized_arr, binary_vals], axis=1), scaler