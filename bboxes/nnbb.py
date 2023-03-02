from tabnanny import verbose
from keras import layers
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from bboxes.bb_wrapper import SklearnClassifierWrapper
import pickle
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

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
    def __init__(self, model=None, scaler=None, db_name=None, regularized=False):
        self.regularized = regularized
        self.model_name = "nn"
        if model:
            self.model = model
            self.scaler = scaler
            self.db_name = db_name
        else:
            self.db_name = db_name
            if regularized:
                self.model = keras.models.load_model("../models/{}/nn/nn_blackbox_regularized.h5".format(db_name))
                self.scaler = pickle.load(open("../models/{}/nn/nn_scaler_regularized.sav".format(db_name), 'rb'))
            else:
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
        elif self.db_name == 'adult':
            x = layers.Dense(300, activation="relu")(inputs)
            x = layers.Dense(300, activation="relu")(x)
            x = layers.Dense(300, activation="relu")(x)
            x = layers.Dense(300, activation="relu")(x)
            x = layers.Dense(300, activation="relu")(x)
            output = layers.Dense(2, activation="softmax")(x)
            opt = tf.optimizers.Adam()
            model = keras.Model(inputs=inputs, outputs=output, name="nn_bb_model")
            model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.fit(tr, tr_l, epochs=250, batch_size=250)
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
    db_name = "adult"
    bb = NeuralNetworkBlackBox(db_name=db_name, regularized=False)
    import pandas as pd
    train_set = pd.read_csv("../data/{}/original_train_set.csv".format(db_name))
    test_set = pd.read_csv("../data/{}/original_test_set.csv".format(db_name))
    train_label = pd.read_csv("../data/{}/original_train_label.csv".format(db_name))
    test_label = pd.read_csv("../data/{}/original_test_label.csv".format(db_name))


    # Performances on training set
    train_prediction = bb.predict(train_set.values)
    train_confidence = bb.predict_proba(train_set.values)
    report = classification_report(train_label, train_prediction)
    print(report)

    # Performances on test set
    test_prediction = bb.predict(test_set.values)
    test_confidence = bb.predict_proba(test_set.values)
    report = classification_report(test_label, test_prediction)
    print(report)

    jointed = list(map(lambda x: np.max(x), test_confidence))
    jointed2 = list(map(lambda x: np.max(x), train_confidence))
    jointed = np.concatenate([jointed, jointed2], axis=0)

    target = np.concatenate([np.ones(test_set.shape[0]), np.zeros(train_set.shape[0])], axis=0)
    df = pd.DataFrame(target, columns=['target'])
    df['confidence'] = jointed
    df.to_csv("./confidence_distrib.csv")

    # create ROC curve
    bb_r = NeuralNetworkBlackBox(db_name=db_name, regularized=True)
    test_prediction_r = bb_r.predict(test_set.values)
    fpr, tpr, _ = metrics.roc_curve(test_label, test_prediction)
    plt.plot(fpr, tpr, label="overfitted")
    fpr, tpr, _ = metrics.roc_curve(test_label, test_prediction_r)
    plt.plot(fpr, tpr, label="regularized")
    plt.plot([0.10 * i for i in range(11)], [0.10 * i for i in range(11)], '--', c="red")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC curve on test")
    plt.legend()
    plt.grid()
    plt.show()

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(test_label, test_prediction_r, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['<=50K', '>50k'])
    disp.plot()
    plt.title("Confution matrix of the NN classifier\n regularized accuracy:{}".format(
        round(metrics.accuracy_score(test_label, test_prediction_r), 3)))
    plt.show()