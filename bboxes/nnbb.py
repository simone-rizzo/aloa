from tabnanny import verbose
from keras import layers
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from bboxes.bb_wrapper import SklearnClassifierWrapper
import pickle
import numpy as np
import tensorflow as tf


def normalize(ds, scaler=None, dataframe=True, db_name='adult'):
    """
    Normalize the dataset in order to be fitted inside the model.
    :param ds: dataframe with the data to be scaled
    :param scaler: if you have already fitted the scaler you can pass it and reuse it.
    :return: scaled dataset
    """
    if db_name == 'adult':
        continuos_val = ds.values[:, :6] if dataframe else ds[:, :6]
        binary_vals = ds.values[:, 6:] if dataframe else ds[:, 6:]
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(continuos_val)
        normalized_arr = scaler.transform(continuos_val)
        return np.concatenate([normalized_arr, binary_vals], axis=1), scaler
    elif db_name == 'bank' or db_name == 'synth':
        ds = ds.values if dataframe else ds
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(ds)
            return  scaler.transform(ds), scaler
        else:
            normalized_arr = scaler.transform(ds)
            return normalized_arr, scaler


class NeuralNetworkBlackBox(SklearnClassifierWrapper):
    def __init__(self, model=None, scaler=None, db_name=None):
        if model:
            self.model = model
            self.scaler = scaler
            self.db_name = db_name
        else:
            self.db_name = db_name
            self.model = keras.models.load_model("../models/{}/nn/nn_blackbox.h5".format(db_name))
            self.scaler = pickle.load(open("../models/{}/nn/nn_scaler.sav".format(db_name), 'rb'))

    def model(self):
        return self.model()

    def predict(self, x):
        x, _ = normalize(x, self.scaler, False, self.db_name) # scaling layer
        out = self.model.predict(x, verbose=False) # feedforward
        out = np.argmax(out, axis=1) # argmax layer
        return out

    def predict_proba(self, x):
        x, _ = normalize(x, self.scaler, False, self.db_name)  # scaling layer
        return self.model.predict(x, verbose=False)

    def train_model(self, tr, tr_l, epochs=100):
        """
        Given the train set and train label, we train a NN
        with the same behaviour as the blackbox.
        """
        tr, _ = normalize(tr, self.scaler, False, db_name=self.db_name)  # scaling layer # first we scale the values.
        inputs = keras.Input(shape=(tr.shape[1],))
        if self.db_name == 'bank':
            x = layers.Dense(300, activation="relu")(inputs)
            x = layers.Dense(300, activation="relu")(x)
            x = layers.Dense(300, activation="relu")(x)
            x = layers.Dense(300, activation="relu")(x)
            x = layers.Dense(300, activation="relu")(x)
            output = layers.Dense(2, activation="softmax")(x)
            opt = tf.optimizers.Adam()
            model = keras.Model(inputs=inputs, outputs=output, name="nn_bb_model")
            model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.fit(tr, tr_l, epochs=epochs, batch_size=250)
            return NeuralNetworkBlackBox(model, self.scaler, self.db_name)
        elif self.db_name == 'synth':
            x = layers.Dense(300, activation="relu")(inputs)
            x = layers.Dense(300, activation="relu")(x)
            x = layers.Dense(300, activation="relu")(x)
            x = layers.Dense(300, activation="relu")(x)
            x = layers.Dense(300, activation="relu")(x)
            output = layers.Dense(15, activation="softmax")(x)
            opt = tf.optimizers.Adam()
            model = keras.Model(inputs=inputs, outputs=output, name="nn_bb_model")
            model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.fit(tr, tr_l, epochs=200, batch_size=1024)
            return NeuralNetworkBlackBox(model, self.scaler, self.db_name)


if __name__ == "__main__":
    db_name = "bank"
    bb = NeuralNetworkBlackBox(db_name=db_name)
    import pandas as pd
    train_set = pd.read_csv("../data/{}/original_train_set.csv".format(db_name))
    test_set = pd.read_csv("../data/{}/original_test_set.csv".format(db_name))
    train_label = pd.read_csv("../data/{}/original_train_label.csv".format(db_name))
    test_label = pd.read_csv("../data/{}/original_test_label.csv".format(db_name))

    # Here we normalize the training set and the test set
    """train_set, scaler = normalize(train_set)
    test_set, _ = normalize(test_set, scaler)"""

    # Performances on training set
    train_prediction = bb.predict(train_set.values)
    report = classification_report(train_label, train_prediction)
    print(report)

    # Performances on test set
    test_prediction = bb.predict(test_set.values)
    report = classification_report(test_label, test_prediction)
    print(report)