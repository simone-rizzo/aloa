from imblearn import keras
from bboxes.bb_wrapper import SklearnClassifierWrapper
import pickle
import numpy as np


class NeuralNetworkBlackBox(SklearnClassifierWrapper):
    def __init__(self):
        target_model = keras.models.load_model("../models/nn/nn_blackbox.h5")
        super().__init__(target_model)

    def model(self):
        return super().model()

    def predict(self, x):
        out = super().predict(x)
        out = np.array(list(map(lambda x: 1 if x > 0.5 else 0, out)))
        return out

    def predict_proba(self, x):
        return super().predict(x)