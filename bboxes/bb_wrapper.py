import abc
from bbox import AbstractBBox
import numpy as np


class SklearnClassifierWrapper(AbstractBBox, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def model(self):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass

    @abc.abstractmethod
    def predict_proba(self, x):
        pass

    @abc.abstractmethod
    def train_model(self, x, y):
        pass
