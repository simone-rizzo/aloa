import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from matplotlib import pyplot as plt


def normalize(ds, scaler=None):
    """
    Normalize the dataset in order to be fitted inside the model.
    :param ds: dataframe with the data to be scaled
    :param scaler: if you have already fitted the scaler you can pass it and reuse it.
    :return: scaled dataset
    """
    continuos_val = ds.values[:, :6]
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(continuos_val)
    normalized_arr = scaler.transform(continuos_val)
    return np.concatenate([normalized_arr, ds.values[:, 6:]], axis=1), scaler


def get_nn_model(input_dim):
    """
    Creation of the neural network for the Adult Task.
    :param input_dim: input dimension.
    :return:
    """
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(300, activation="tanh")(inputs)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(300, activation="tanh")(x)
    x = layers.Dense(300, activation="tanh")(x)
    x = layers.Dense(300, activation="tanh")(x)
    x = layers.Dense(300, activation="tanh")(x)
    x = layers.Dense(300, activation="tanh")(x)
    x = layers.Dense(300, activation="tanh")(x)
    # x = layers.Dropout(0.1)(x)
    # output = layers.Dense(1, activation="sigmoid")(x)
    output = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=output, name="nn_bb_model")
    return model


def load_nn_bb(filepath):
    from tensorflow import keras
    return keras.models.load_model(filepath)


ds_name = "adult"
train_set = pd.read_csv("../../../data/{}/original_train_set.csv".format(ds_name))
test_set = pd.read_csv("../../../data/{}/original_test_set.csv".format(ds_name))
train_label = pd.read_csv("../../../data/{}/original_train_label.csv".format(ds_name))
test_label = pd.read_csv("../../../data/{}/original_test_label.csv".format(ds_name))

# Here we normalize the training set and the test set
train_set, scaler = normalize(train_set)
test_set, _ = normalize(test_set, scaler)
pickle.dump(scaler, open("nn_scaler_regularized.sav", 'wb'))
# scaler = pickle.load(open(scalerfile, 'rb'))

# Creation of the model
model = get_nn_model(train_set.shape[1])

# Undersampling of the dataset
undersample = RandomUnderSampler(sampling_strategy="majority")
tr, tr_l = undersample.fit_resample(train_set, train_label.values)

# Compilation of the model and training.
opt = tf.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(tr, tr_l, epochs=250, batch_size=512)

# Performances on training set
train_prediction = model.predict(tr)
train_prediction = np.argmax(train_prediction, axis=1)
report = classification_report(tr_l, train_prediction)
print(report)

# Performances on test set
test_prediction = model.predict(test_set)
test_prediction = np.argmax(test_prediction, axis=1)
report = classification_report(test_label, test_prediction)
print(report)

# Saving the model
model.save('nn_blackbox_regularized.h5')

plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
