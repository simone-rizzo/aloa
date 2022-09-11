from tabnanny import verbose
from keras import layers
from tensorflow import keras
from bboxes.bb_wrapper import SklearnClassifierWrapper
import pickle
import numpy as np
import tensorflow as tf


class NeuralNetworkBlackBox(SklearnClassifierWrapper):
    def __init__(self, model=None):
        if model:
            self.model = model
        else:
            self.model = keras.models.load_model("models/nn/nn_blackbox.h5")

    def model(self):
        return self.model()

    def predict(self, x):
        out = self.model.predict(x, verbose=False)
        out = np.argmax(out, axis=1)
        return out

    def predict_proba(self, x):
        return self.model.predict(x, verbose=False)

    def train_model(self, tr, tr_l):
        """
        Given the train set and train label, we train a NN
        with the same behaviour as the blackbox.
        """
        inputs = keras.Input(shape=(tr.shape[1],))
        x = layers.Dense(256, activation="tanh")(inputs)
        # x = layers.Dropout(0.1)(x)
        x = layers.Dense(256, activation="tanh")(x)
        # x = layers.Dropout(0.1)(x)
        output = layers.Dense(2, activation="softmax")(x)
        opt = tf.optimizers.Adam(learning_rate=0.005)
        model = keras.Model(inputs=inputs, outputs=output, name="nn_bb_model")
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.fit(tr, tr_l, epochs=400, batch_size=16)
        return NeuralNetworkBlackBox(model)


if __name__ == "__main__":
    bb = NeuralNetworkBlackBox()
    print(bb.model().summary())