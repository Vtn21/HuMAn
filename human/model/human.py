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
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402
from tensorflow.keras.layers.experimental import preprocessing  # noqa: E402
from tensorflow.keras.regularizers import L2  # noqa: E402


L2_PENALTY = 0.001
NPZ_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(),
                        "normalization.npz")


def prediction_subnet(selection, inputs, name="joint"):
    conc = layers.Concatenate(axis=2, name=f"{name}_concat")(inputs)
    dense1 = layers.Dense(64, activation="tanh", name=f"{name}_dense1",
                          kernel_regularizer=L2(L2_PENALTY))(conc)
    dropout1 = layers.Dropout(0.2, name=f"{name}_dropout1")(dense1)
    dense2 = layers.Dense(32, activation="tanh", name=f"{name}_dense2",
                          kernel_regularizer=L2(L2_PENALTY))(dropout1)
    dropout2 = layers.Dropout(0.2, name=f"{name}_dropout2")(dense2)
    dense3 = layers.Dense(32, activation="tanh", name=f"{name}_dense3",
                          kernel_regularizer=L2(L2_PENALTY))(dropout2)
    dropout3 = layers.Dropout(0.2, name=f"{name}_dropout3")(dense3)
    linear = layers.Dense(3, name=f"{name}_linear")(dropout3)
    return layers.Multiply(name=f"{name}_multiply")([linear, selection])


def normalization_layer(input_data):
    if isinstance(input_data, tf.data.Dataset):
        normalization = preprocessing.Normalization(axis=2,
                                                    name="normalization")
        normalization.adapt(input_data)
        # Save mean and variance to npz file, for later usage
        np.savez(NPZ_PATH, mean=normalization.mean.numpy(),
                 variance=normalization.variance.numpy())
        return normalization
    elif isinstance(input_data, str):
        n = np.load(input_data)
        mean = tf.convert_to_tensor(n["mean"])
        variance = tf.convert_to_tensor(n["variance"])
        normalization = preprocessing.Normalization(axis=2, mean=mean,
                                                    variance=variance,
                                                    name="normalization")
        return normalization
    else:
        raise ValueError("Invalid input data provided for normalization")


def get_human_model(pose_input_dataset=None, normalization_npz=NPZ_PATH):
    # Input layers
    pose_input = keras.Input(shape=(None, 72), name="pose_input")
    selection_input = keras.Input(shape=(None, 72), name="selection_input")
    time_input = keras.Input(shape=(None, 1), name="time_input")
    # Normalize the pose inputs
    try:
        # Try to load the specified npz file and create the normalization layer
        normalization = normalization_layer(normalization_npz)
    except Exception as ex:
        print(f"Unable to load {normalization_npz}: {ex}")
        # Try to use the pose dataset instead
        if pose_input_dataset is not None:
            print("Falling back to provided dataset")
            normalization = normalization_layer(pose_input_dataset)
        else:
            raise ValueError("No dataset provided for fallback; unable to\
                             create normalization layer.")
    # Apply the normalization layer to the pose input
    pose_norm = normalization(pose_input)
    # Discard some joints using the selection input
    pose_norm_select = layers.Multiply(name="pose_norm_select")(
        [pose_norm, selection_input])
    # Dropout on pose input (avoid overfitting)
    pose_dropout = layers.Dropout(0.1, name="pose_dropout")(pose_norm_select)
    # Concatenate pose and time inputs
    concat_inputs = layers.Concatenate(axis=2, name="inputs_concat")(
        [pose_dropout, time_input])
    # LSTM layer with dropout
    seq_output = layers.LSTM(1024, return_sequences=True, name="LSTM",
                             kernel_regularizer=L2(L2_PENALTY))(concat_inputs)
    seq_dropout = layers.Dropout(0.2, name="LSTM_dropout")(seq_output)
    # Prediction layers
    delta = []
    # Prediction layer: root orientation
    s0 = tf.gather(selection_input, [0, 1, 2], axis=2)
    delta.append(prediction_subnet(s0, [seq_dropout, time_input],
                                   name="root"))
    # Prediction layer: left hip
    s1 = tf.gather(selection_input, [3, 4, 5], axis=2)
    delta.append(prediction_subnet(s1, [seq_dropout, time_input,
                                        delta[0]], name="hipL"))
    # Prediction layer: right hip
    s2 = tf.gather(selection_input, [6, 7, 8], axis=2)
    delta.append(prediction_subnet(s2, [seq_dropout, time_input,
                                        delta[0]], name="hipR"))
    # Prediction layer: lower spine
    s3 = tf.gather(selection_input, [9, 10, 11], axis=2)
    delta.append(prediction_subnet(s3, [seq_dropout, time_input,
                                        delta[0]], name="spine3"))
    # Prediction layer: left knee
    s4 = tf.gather(selection_input, [12, 13, 14], axis=2)
    delta.append(prediction_subnet(s4, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 1]],
                                   name="kneeL"))
    # Prediction layer: right knee
    s5 = tf.gather(selection_input, [15, 16, 17], axis=2)
    delta.append(prediction_subnet(s5, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 2]],
                                   name="kneeR"))
    # Prediction layer: mid spine
    s6 = tf.gather(selection_input, [18, 19, 20], axis=2)
    delta.append(prediction_subnet(s6, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3]],
                                   name="spine6"))
    # Prediction layer: left ankle
    s7 = tf.gather(selection_input, [21, 22, 23], axis=2)
    delta.append(prediction_subnet(s7, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 1, 4]],
                                   name="ankleL"))
    # Prediction layer: right ankle
    s8 = tf.gather(selection_input, [24, 25, 26], axis=2)
    delta.append(prediction_subnet(s8, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 2, 5]],
                                   name="ankleR"))
    # Prediction layer: high spine
    s9 = tf.gather(selection_input, [27, 28, 29], axis=2)
    delta.append(prediction_subnet(s9, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6]],
                                   name="spine9"))
    # Prediction layer: left foot
    s10 = tf.gather(selection_input, [30, 31, 32], axis=2)
    delta.append(prediction_subnet(s10, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 1, 4, 7]],
                                   name="footL"))
    # Prediction layer: right foot
    s11 = tf.gather(selection_input, [33, 34, 35], axis=2)
    delta.append(prediction_subnet(s11, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 2, 5, 8]],
                                   name="footR"))
    # Prediction layer: neck
    s12 = tf.gather(selection_input, [36, 37, 38], axis=2)
    delta.append(prediction_subnet(s12, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9]],
                                   name="neck"))
    # Prediction layer: left clavicula
    s13 = tf.gather(selection_input, [39, 40, 41], axis=2)
    delta.append(prediction_subnet(s13, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9]],
                                   name="clavL"))
    # Prediction layer: right clavicula
    s14 = tf.gather(selection_input, [42, 43, 44], axis=2)
    delta.append(prediction_subnet(s14, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9]],
                                   name="clavR"))
    # Prediction layer: head
    s15 = tf.gather(selection_input, [45, 46, 47], axis=2)
    delta.append(prediction_subnet(s15, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9, 12]],
                                   name="head"))
    # Prediction layer: left shoulder
    s16 = tf.gather(selection_input, [48, 49, 50], axis=2)
    delta.append(prediction_subnet(s16, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9, 13]],
                                   name="shoulderL"))
    # Prediction layer: right shoulder
    s17 = tf.gather(selection_input, [51, 52, 53], axis=2)
    delta.append(prediction_subnet(s17, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9, 14]],
                                   name="shoulderR"))
    # Prediction layer: left elbow
    s18 = tf.gather(selection_input, [54, 55, 56], axis=2)
    delta.append(prediction_subnet(s18, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9, 13, 16]],
                                   name="elbowL"))
    # Prediction layer: right elbow
    s19 = tf.gather(selection_input, [57, 58, 59], axis=2)
    delta.append(prediction_subnet(s19, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9, 14, 17]],
                                   name="elbowR"))
    # Prediction layer: left wrist
    s20 = tf.gather(selection_input, [60, 61, 62], axis=2)
    delta.append(prediction_subnet(s20, [seq_dropout, time_input] + [delta[i]
                                   for i in [0, 3, 6, 9, 13, 16, 18]],
                                   name="wristL"))
    # Prediction layer: right wrist
    s21 = tf.gather(selection_input, [63, 64, 65], axis=2)
    delta.append(prediction_subnet(s21, [seq_dropout, time_input] + [delta[i]
                                   for i in [0, 3, 6, 9, 14, 17, 19]],
                                   name="wristR"))
    # Prediction layer: left hand
    s22 = tf.gather(selection_input, [66, 67, 68], axis=2)
    delta.append(prediction_subnet(s22, [seq_dropout, time_input] + [delta[i]
                                   for i in [0, 3, 6, 9, 13, 16, 18, 20]],
                                   name="handL"))
    # Prediction layer: right hand
    s23 = tf.gather(selection_input, [69, 70, 71], axis=2)
    delta.append(prediction_subnet(s23, [seq_dropout, time_input] + [delta[i]
                                   for i in [0, 3, 6, 9, 14, 17, 19, 21]],
                                   name="handR"))
    # Concatenate all predictions
    deltas = layers.Concatenate(axis=2)(delta)
    # Final prediction: pose_input (selected) + deltas
    pose_select = layers.Multiply(name="pose_select")(
        [pose_input, selection_input])
    pose_pred = layers.Add()([pose_select, deltas])
    # Create the model
    model = keras.Model(inputs=[pose_input, selection_input, time_input],
                        outputs=[pose_pred])
    return model


if __name__ == "__main__":
    model = get_human_model()
    keras.utils.plot_model(model, show_shapes=True)
    model.summary()
