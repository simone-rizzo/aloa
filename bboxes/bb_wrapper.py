import abc
from bbox import AbstractBBox
import numpy as np


class SklearnClassifierWrapper(AbstractBBox, metaclass=abc.ABCMeta):
    def __init__(self, classifier):
        super().__init__()
        self.bbox = classifier

    def model(self):
        return self.bbox

    def predict(self, x):
        return self.bbox.predict(x)

    def predict_proba(self, x):
        return self.bbox.predict_proba(x)

    @abc.abstractmethod
    def train_model(self, x, y):
        pass
