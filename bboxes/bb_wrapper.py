import abc
from bboxes.bbox import AbstractBBox


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
