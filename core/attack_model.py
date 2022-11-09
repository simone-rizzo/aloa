from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np


class AttackModel:
    def __init__(self, data, target, attack_type="confidence"):
        inputs = keras.Input(shape=(data.shape[1],), )
        x = layers.Dense(100, activation="relu")(inputs)
        x = layers.Dense(100, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        out = layers.Dense(2, activation="softmax")(x)
        mdl = keras.Model(inputs=inputs, outputs=out, name="nn_bb_model")
        opt = tf.optimizers.Adam()
        mdl.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        epocs_dict = {'confidence': 200, 'perturb': 10}
        mdl.fit(x=data, y=target, epochs=epocs_dict[attack_type], batch_size=128)
        self.model = mdl

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def predict_proba(self, x):
        return self.model.predict(x, verbose=False)