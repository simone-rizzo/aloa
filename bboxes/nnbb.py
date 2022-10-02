from tabnanny import verbose
from keras import layers
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from bboxes.bb_wrapper import SklearnClassifierWrapper
import pickle
import numpy as np
import tensorflow as tf


def normalize(ds, scaler=None, dataframe=True):
    """
    Normalize the dataset in order to be fitted inside the model.
    :param ds: dataframe with the data to be scaled
    :param scaler: if you have already fitted the scaler you can pass it and reuse it.
    :return: scaled dataset
    """
    continuos_val = ds.values[:, :6] if dataframe else ds[:, :6]
    binary_vals = ds.values[:, 6:] if dataframe else ds[:, 6:]
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(continuos_val)
    normalized_arr = scaler.transform(continuos_val)
    return np.concatenate([normalized_arr, binary_vals], axis=1), scaler


class NeuralNetworkBlackBox(SklearnClassifierWrapper):
    def __init__(self, model=None):
        if model:
            self.model = model
        else:
            self.model = keras.models.load_model("models/nn/nn_blackbox_regolarized.h5")

    def model(self):
        return self.model()

    def predict(self, x):
        out = self.model.predict(x, verbose=False)
        out = np.argmax(out, axis=1)
        return out

    def predict_proba(self, x):
        return self.model.predict(x, verbose=False)

    def train_model(self, tr, tr_l, epochs=400):
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
        model.fit(tr, tr_l, epochs=epochs, batch_size=16)
        return NeuralNetworkBlackBox(model)


if __name__ == "__main__":
    bb = NeuralNetworkBlackBox()

    import pandas as pd
    train_set = pd.read_csv("data/adult_original_train_set.csv")
    test_set = pd.read_csv("data/adult_original_test_set.csv")
    train_label = pd.read_csv("data/adult_original_train_label.csv")
    test_label = pd.read_csv("data/adult_original_test_label.csv")

    # Here we normalize the training set and the test set
    train_set, scaler = normalize(train_set)
    test_set, _ = normalize(test_set, scaler)

    # Performances on training set
    train_prediction = bb.predict(train_set)
    report = classification_report(train_label, train_prediction)
    print(report)

    # Performances on test set
    test_prediction = bb.predict(test_set)
    report = classification_report(test_label, test_prediction)
    print(report)