import abc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from bboxes.rfbb import RandomForestBlackBox


class Attack(metaclass=abc.ABCMeta):

    def trainRFClassifier(self, x, y):
        rf = RandomForestClassifier()
        rf.fit(x, y)
        return rf

    @abc.abstractmethod
    def attack_workflow(self):
        pass

    def start_attack(self):
        self.initialize_dataset()
        self.split_noise_dataset()
        self.attack_workflow()

    def initialize_dataset(self):
        self.noise_train_set = pd.read_csv("../data/adult_noise_shadow_labelled")
        self.noise_train_label = self.noise_train_set.pop("class")
        self.train_set = pd.read_csv("../data/adult_original_train_set.csv")
        self.train_label = pd.read_csv("../data/adult_original_train_label.csv")
        self.test_set = pd.read_csv("../data/adult_original_test_set.csv")
        self.test_label = pd.read_csv("../data/adult_original_test_label.csv")
        self.bb = RandomForestBlackBox()

    def split_noise_dataset(self):
        self.noise_train_set, self.noise_test_set, self.noise_train_label, self.noise_test_label = train_test_split(
            self.noise_train_set, self.noise_train_label, stratify=self.noise_train_label,
            test_size=0.2, random_state=1)