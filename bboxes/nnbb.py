from keras import layers
from tensorflow import keras
from bboxes.bb_wrapper import SklearnClassifierWrapper
import pickle
import numpy as np
import tensorflow as tf


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

    def train_model(self, x, y):
        """
        Given the train set and train label, we train a NN
        with the same behaviour as the blackbox.
        """
        inputs = keras.Input(shape=(x.shape[1],))
        x = layers.Dense(256, activation="tanh")(inputs)
        # x = layers.Dropout(0.1)(x)
        x = layers.Dense(256, activation="tanh")(x)
        # x = layers.Dropout(0.1)(x)
        output = layers.Dense(1, activation="sigmoid")(x)
        opt = tf.optimizers.Adam(learning_rate=0.005)
        model = keras.Model(inputs=inputs, outputs=output, name="nn_bb_model")
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
        model.fit(x, y, epochs=30, batch_size=16)
        return model


if __name__ == "__main__":
    bb = NeuralNetworkBlackBox()
    print(bb.model().summary())