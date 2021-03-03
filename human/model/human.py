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

    def __init__(self, units=(64, 32), rate=0.2, name="joint"):
        super().__init__()
        self.concatenate = layers.Concatenate(axis=2, name=f"{name}_concat")
        self.dense0 = layers.Dense(units[0], activation="tanh",
                                   name=f"{name}_dense0",
                                   kernel_regularizer=L2(L2_PENALTY))
        self.dropout0 = layers.Dropout(rate, name=f"{name}_dropout0")
        self.dense1 = layers.Dense(units[1], activation="tanh",
                                   name=f"{name}_dense1",
                                   kernel_regularizer=L2(L2_PENALTY))
        self.linear = layers.Dense(3, name=f"{name}_linear")
        self.dropout1 = layers.Dropout(rate, name=f"{name}_dropout1")
        self.multiply = layers.Multiply(name=f"{name}_multiply")

    def call(self, inputs, training=False):
        """Forward pass of the sub-network.

        Args:
            inputs (list of tensors): must receive the selection input
                (1x3 array with zeros or ones) as the first input, together
                with the time input, the LSTM output after dropout, and the
                predictions from the parent joints.

        Returns:
            (1x3 tensor): displacement prediction for this specific joint.
        """
        x = self.concatenate(inputs[1:])
        x = self.dense0(x, training=training)
        x = self.dropout0(x, training=training)
        x = self.dense1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.linear(x, training=training)
        return self.multiply([inputs[0], x])


class HuMAn(tf.keras.Model):
    def __init__(self, norm_dataset=None, norm_npz=NPZ_PATH,
                 lstm_units=1024, rate=(0.1, 0.2)):
        super().__init__()
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
        self.root = JointPredictionLayer(name="root")
        self.hipL = JointPredictionLayer(name="hipL")
        self.hipR = JointPredictionLayer(name="hipR")
        self.spine3 = JointPredictionLayer(name="spine3")
        self.kneeL = JointPredictionLayer(name="kneeL")
        self.kneeR = JointPredictionLayer(name="kneeR")
        self.spine6 = JointPredictionLayer(name="spine6")
        self.ankleL = JointPredictionLayer(name="ankleL")
        self.ankleR = JointPredictionLayer(name="ankleR")
        self.spine9 = JointPredictionLayer(name="spine9")
        self.footL = JointPredictionLayer(name="footL")
        self.footR = JointPredictionLayer(name="footR")
        self.neck = JointPredictionLayer(name="neck")
        self.clavL = JointPredictionLayer(name="clavL")
        self.clavR = JointPredictionLayer(name="clavR")
        self.head = JointPredictionLayer(name="head")
        self.shldrL = JointPredictionLayer(name="shldrL")
        self.shldrR = JointPredictionLayer(name="shldrR")
        self.elbowL = JointPredictionLayer(name="elbowL")
        self.elbowR = JointPredictionLayer(name="elbowR")
        self.wristL = JointPredictionLayer(name="wristL")
        self.wristR = JointPredictionLayer(name="wristR")
        self.handL = JointPredictionLayer(name="handL")
        self.handR = JointPredictionLayer(name="handR")
        # Output layers
        self.concat_deltas = layers.Concatenate(axis=2, name="concat_deltas")
        self.add = layers.Add()

    def call(self, inputs, state=None, return_state=False, training=False):
        # Parsing inputs
        pose_input = inputs[0]
        selection_input = inputs[1]
        time_input = inputs[2]
        # Forward pass
        x = self.normalization(pose_input)
        x = self.pose_select([x, selection_input])
        x = self.dropout_input(x, training=training)
        x = self.concat_inputs([x, time_input])
        # LSTM layer
        x, *state = self.lstm(x, initial_state=state, training=training)
        x = self.dropout_lstm(x, training=training)
        # Prediction layers
        delta_root = self.root([tf.gather(selection_input, [0, 1, 2], axis=2),
                                time_input, x], training=training)
        delta_hipL = self.hipL([tf.gather(selection_input, [3, 4, 5], axis=2),
                                time_input, x, delta_root], training=training)
        delta_hipR = self.hipR([tf.gather(selection_input, [6, 7, 8], axis=2),
                                time_input, x, delta_root], training=training)
        delta_spine3 = self.spine3([tf.gather(selection_input, [9, 10, 11],
                                              axis=2), time_input, x,
                                    delta_root], training=training)
        delta_kneeL = self.kneeL([tf.gather(selection_input, [12, 13, 14],
                                            axis=2), time_input, x,
                                  delta_root, delta_hipL], training=training)
        delta_kneeR = self.kneeR([tf.gather(selection_input, [15, 16, 17],
                                            axis=2), time_input, x,
                                  delta_root, delta_hipR], training=training)
        delta_spine6 = self.spine6([tf.gather(selection_input, [18, 19, 20],
                                              axis=2), time_input, x,
                                    delta_root, delta_spine3],
                                   training=training)
        delta_ankleL = self.ankleL([tf.gather(selection_input, [21, 22, 23],
                                              axis=2), time_input, x,
                                    delta_root, delta_hipL, delta_kneeL],
                                   training=training)
        delta_ankleR = self.ankleR([tf.gather(selection_input, [24, 25, 26],
                                              axis=2), time_input, x,
                                    delta_root, delta_hipR, delta_kneeR],
                                   training=training)
        delta_spine9 = self.spine9([tf.gather(selection_input, [27, 28, 29],
                                              axis=2), time_input, x,
                                    delta_root, delta_spine3, delta_spine6],
                                   training=training)
        delta_footL = self.footL([tf.gather(selection_input, [30, 31, 32],
                                            axis=2), time_input, x,
                                  delta_root, delta_hipL, delta_kneeL,
                                  delta_ankleL], training=training)
        delta_footR = self.footR([tf.gather(selection_input, [33, 34, 35],
                                            axis=2), time_input, x,
                                  delta_root, delta_hipR, delta_kneeR,
                                  delta_ankleR], training=training)
        delta_neck = self.neck([tf.gather(selection_input, [36, 37, 38],
                                          axis=2), time_input, x,
                                delta_root, delta_spine3, delta_spine6,
                                delta_spine9], training=training)
        delta_clavL = self.clavL([tf.gather(selection_input, [39, 40, 41],
                                            axis=2), time_input, x,
                                  delta_root, delta_spine3, delta_spine6,
                                  delta_spine9], training=training)
        delta_clavR = self.clavR([tf.gather(selection_input, [42, 43, 44],
                                            axis=2), time_input, x,
                                  delta_root, delta_spine3, delta_spine6,
                                  delta_spine9], training=training)
        delta_head = self.head([tf.gather(selection_input, [45, 46, 47],
                                          axis=2), time_input, x,
                                delta_root, delta_spine3, delta_spine6,
                                delta_spine9, delta_neck], training=training)
        delta_shldrL = self.shldrL([tf.gather(selection_input, [48, 49, 50],
                                              axis=2), time_input, x,
                                    delta_root, delta_spine3, delta_spine6,
                                    delta_spine9, delta_clavL],
                                   training=training)
        delta_shldrR = self.shldrR([tf.gather(selection_input, [51, 52, 53],
                                              axis=2), time_input, x,
                                    delta_root, delta_spine3, delta_spine6,
                                    delta_spine9, delta_clavR],
                                   training=training)
        delta_elbowL = self.elbowL([tf.gather(selection_input, [54, 55, 56],
                                              axis=2), time_input, x,
                                    delta_root, delta_spine3, delta_spine6,
                                    delta_spine9, delta_clavL, delta_shldrL],
                                   training=training)
        delta_elbowR = self.elbowR([tf.gather(selection_input, [57, 58, 59],
                                              axis=2), time_input, x,
                                    delta_root, delta_spine3, delta_spine6,
                                    delta_spine9, delta_clavR, delta_shldrR],
                                   training=training)
        delta_wristL = self.wristL([tf.gather(selection_input, [60, 61, 62],
                                              axis=2), time_input, x,
                                    delta_root, delta_spine3, delta_spine6,
                                    delta_spine9, delta_clavL, delta_shldrL,
                                    delta_elbowL], training=training)
        delta_wristR = self.wristR([tf.gather(selection_input, [63, 64, 65],
                                              axis=2), time_input, x,
                                    delta_root, delta_spine3, delta_spine6,
                                    delta_spine9, delta_clavR, delta_shldrR,
                                    delta_elbowR], training=training)
        delta_handL = self.handL([tf.gather(selection_input, [66, 67, 68],
                                            axis=2), time_input, x,
                                  delta_root, delta_spine3, delta_spine6,
                                  delta_spine9, delta_clavL, delta_shldrL,
                                  delta_elbowL, delta_wristL],
                                 training=training)
        delta_handR = self.handR([tf.gather(selection_input, [69, 70, 71],
                                            axis=2), time_input, x,
                                  delta_root, delta_spine3, delta_spine6,
                                  delta_spine9, delta_clavR, delta_shldrR,
                                  delta_elbowR, delta_wristR],
                                 training=training)
        # Concatenate
        deltas = self.concat_deltas([delta_root, delta_hipL, delta_hipR,
                                     delta_spine3, delta_kneeL, delta_kneeR,
                                     delta_spine6, delta_ankleL, delta_ankleR,
                                     delta_spine9, delta_footL, delta_footR,
                                     delta_neck, delta_clavL, delta_clavR,
                                     delta_head, delta_shldrL, delta_shldrR,
                                     delta_elbowL, delta_elbowR, delta_wristL,
                                     delta_wristR, delta_handL, delta_handR])
        # Residual connection
        preds = self.add([self.pose_select([pose_input, selection_input]),
                          deltas])
        # Return
        if return_state:
            return preds, state
        else:
            return preds


if __name__ == "__main__":
    pose_input = tf.keras.Input(shape=(None, 72))
    selection_input = tf.keras.Input(shape=(None, 72))
    time_input = tf.keras.Input(shape=(None, 1))
    model = HuMAn()
    model([pose_input, selection_input, time_input])
    model.summary()
