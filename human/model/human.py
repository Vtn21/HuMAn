"""
human.py

Defines the HuMAn neural network model.
Running this script plots a graphical representation to "model.png".

Author: Victor T. N.
"""


import numpy as np
import os
import pathlib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers  # noqa: E402
from tensorflow.keras.layers.experimental import preprocessing  # noqa: E402
from tensorflow.keras.regularizers import L2  # noqa: E402


L2_PENALTY = 0.001
NPZ_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(),
                        "normalization.npz")


class JointPredictionLayer(tf.keras.layers.Layer):
    """Implements the sub-network used for predicting future motion of
    a single joint (3-DoF).
    """

    def __init__(self, units=[64, 32, 32], rate=0.2, name="joint"):
        super().__init__()
        self.concatenate = layers.Concatenate(axis=2, name=f"{name}_concat")
        self.dense0 = layers.Dense(units[0], activation="tanh",
                                   name=f"{name}_dense0",
                                   kernel_regularizer=L2(L2_PENALTY))
        self.dropout0 = layers.Dropout(rate, name=f"{name}_dropout0")
        self.dense1 = layers.Dense(units[1], activation="tanh",
                                   name=f"{name}_dense1",
                                   kernel_regularizer=L2(L2_PENALTY))
        self.dropout1 = layers.Dropout(rate, name=f"{name}_dropout1")
        self.dense2 = layers.Dense(units[2], activation="tanh",
                                   name=f"{name}_dense2",
                                   kernel_regularizer=L2(L2_PENALTY))
        self.dropout2 = layers.Dropout(rate, name=f"{name}_dropout2")
        self.linear = layers.Dense(3, name=f"{name}_linear")
        self.multiply = layers.Multiply(name=f"{name}_multiply")

    def call(self, selection_input, elapsed_input, horizon_input, lstm_dropout,
             parent_preds=[], training=False):
        """Forward pass of the sub-network.

        Args:
            inputs (list of tensors): must receive the selection input
                (1x3 array with zeros or ones) as the first input, together
                with the elapsed time input, the time horizon input, the LSTM
                output after dropout, and the predictions from the parent
                joints.

        Returns:
            (1x3 tensor): displacement prediction for this specific joint.
        """
        x = self.concatenate([elapsed_input, horizon_input, lstm_dropout] +
                             parent_preds)
        x = self.dense0(x, training=training)
        x = self.dropout0(x, training=training)
        x = self.dense1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.linear(x, training=training)
        return self.multiply([selection_input, x])


class HuMAn(tf.keras.Model):
    def __init__(self, norm_dataset=None, norm_npz=NPZ_PATH,
                 lstm_units=1024, rate=[0.1, 0.2]):
        super().__init__()
        # Kinematic tree
        self.joints = ["root", "hipL", "hipR", "spine3", "kneeL", "kneeR",
                       "spine6", "ankleL", "ankleR", "spine9", "footL",
                       "footR", "neck", "clavL", "clavR", "head", "shldrL",
                       "shldrR", "elbowL", "elbowR", "wristL", "wristR",
                       "handL", "handR"]
        self.parent = [[], [0], [0], [0], [0, 1], [0, 2], [0, 3], [0, 1, 4],
                       [0, 2, 5], [0, 3, 6], [0, 1, 4, 7], [0, 2, 5, 8],
                       [0, 3, 6, 9], [0, 3, 6, 9], [0, 3, 6, 9],
                       [0, 3, 6, 9, 12], [0, 3, 6, 9, 13], [0, 3, 6, 9, 14],
                       [0, 3, 6, 9, 13, 16], [0, 3, 6, 9, 14, 17],
                       [0, 3, 6, 9, 13, 16, 18], [0, 3, 6, 9, 14, 17, 19],
                       [0, 3, 6, 9, 13, 16, 18, 20],
                       [0, 3, 6, 9, 14, 17, 19, 21]]
        # Normalization layer
        try:
            # Try to load mean and variance from npz path
            loaded_npz = np.load(norm_npz)
        except (IOError, ValueError) as ex:
            # Error loading the npz file
            print(f"Unable to load {norm_npz}: {ex}")
            if norm_dataset is not None:
                # Try to use a provided dataset to adapt the layer
                print("Falling back to provided dataset.")
                self.normalization = preprocessing.Normalization(
                    axis=2, name="normalization")
                print("Adapting layer...")
                self.normalization.adapt(norm_dataset)
                # Save mean and variance for later usage
                print(f"Done! Saving mean and variance to {norm_npz}")
                np.savez(norm_npz, mean=self.normalization.mean.numpy(),
                         variance=self.normalization.variance.numpy())
            else:
                raise ValueError("No dataset provided for fallback; "
                                 "unable to create normalization layer.")
        else:
            # Successfully loaded npz file
            mean = tf.convert_to_tensor(loaded_npz["mean"])
            variance = tf.convert_to_tensor(loaded_npz["variance"])
            self.normalization = preprocessing.Normalization(
                    axis=2, mean=mean, variance=variance, name="normalization")
        # Initial layers
        self.pose_select = layers.Multiply(name="pose_select")
        self.dropout_input = layers.Dropout(rate[0], name="dropout")
        self.concat_inputs = layers.Concatenate(axis=2, name="concat_inputs")
        self.lstm = layers.LSTM(lstm_units, return_sequences=True,
                                return_state=True, name="lstm",
                                kernel_regularizer=L2(L2_PENALTY))
        self.dropout_lstm = layers.Dropout(rate[1], name="dropout_lstm")
        # Prediction layers
        self.pred = []
        for name in self.joints:
            self.pred.append(JointPredictionLayer(name=name))
        # Output layers
        self.concat_deltas = layers.Concatenate(axis=2, name="concat_deltas")
        self.add = layers.Add()

    def call(self, inputs, state=None, return_state=False, training=False):
        # Parsing inputs
        pose_input = inputs["pose_input"]
        selection_input = inputs["selection_input"]
        elapsed_input = inputs["elapsed_input"]
        horizon_input = inputs["horizon_input"]
        # Forward pass
        x = self.normalization(pose_input)
        x = self.pose_select([x, selection_input])
        x = self.dropout_input(x, training=training)
        x = self.concat_inputs([x, elapsed_input])
        # LSTM layer
        x, *state = self.lstm(x, initial_state=state, training=training)
        x = self.dropout_lstm(x, training=training)
        # Prediction layers
        delta = []
        for i, parent in enumerate(self.parent):
            delta.append(self.pred[i](tf.gather(selection_input,
                                                [3*i, 3*i+1, 3*i+2], axis=2),
                                      elapsed_input, horizon_input, x,
                                      [delta[p] for p in parent],
                                      training=training))
        # Residual connection
        preds = self.add([self.pose_select([pose_input, selection_input]),
                          self.concat_deltas(delta)])
        # Return
        if return_state:
            return preds, state
        else:
            return preds


if __name__ == "__main__":
    inputs = {"pose_input": tf.keras.Input(shape=(None, 72)),
              "selection_input": tf.keras.Input(shape=(None, 72)),
              "elapsed_input": tf.keras.Input(shape=(None, 1)),
              "horizon_input": tf.keras.Input(shape=(None, 1))}
    model = HuMAn()
    model(inputs)
    model.summary()
