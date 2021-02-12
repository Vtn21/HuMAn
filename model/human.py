"""
human.py

Defines the HuMAn neural network model.

Author: Victor T. N.
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402


def get_prediction_subnet(inputs):
    conc = layers.Concatenate(axis=2)(inputs)
    dense1 = layers.Dense(128, activation="relu")(conc)
    dense2 = layers.Dense(64, activation="relu")(dense1)
    return layers.Dense(3)(dense2)


def get_human_model():
    # Input layers
    pose_input = keras.Input(shape=(None, 72), name="pose_input")
    selection_input = keras.Input(shape=(None, 72), name="selection_input")
    time_input = keras.Input(shape=(None, 1), name="time_input")
    # Normalize the pose inputs
    pose_norm = layers.BatchNormalization(axis=2)(pose_input)
    # Dropout on pose input (avoid overfitting)
    pose_dropout = layers.Dropout(0.1, name="pose_dropout")(pose_norm)
    # Concatenate all inputs
    concat_inputs = layers.Concatenate(axis=2, name="concat_inputs")(
        [pose_dropout, selection_input, time_input])
    # LSTM layer with dropout
    seq_output = layers.LSTM(2048, return_sequences=True, name="LSTM")(
        concat_inputs)
    seq_dropout = layers.Dropout(0.2, name="LSTM_dropout")(seq_output)
    # Prediction layers
    neurons = 128
    delta = []
    # Prediction layer: root orientation
    s0 = selection_input[:, :, :3]
    linear0 = get_prediction_subnet([seq_dropout, s0, time_input])
    delta.append(layers.Multiply()([linear0, s0]))
    # Prediction layer: left hip
    s1 = selection_input[:, :, 3:6]
    linear1 = get_prediction_subnet([seq_dropout, s1, time_input,
                                     delta[0]])
    delta.append(layers.Multiply()([linear1, s1]))
    # Prediction layer: right hip
    s2 = selection_input[:, :, 6:9]
    linear2 = get_prediction_subnet([seq_dropout, s2, time_input,
                                     delta[0]])
    delta.append(layers.Multiply()([linear2, s2]))
    # Prediction layer: lower spine
    s3 = selection_input[:, :, 9:12]
    linear3 = get_prediction_subnet([seq_dropout, s3, time_input,
                                     delta[0]])
    delta.append(layers.Multiply()([linear3, s3]))
    # Prediction layer: left knee
    s4 = selection_input[:, :, 12:15]
    linear4 = get_prediction_subnet([seq_dropout, s4, time_input,
                                     delta[0], delta[1]])
    delta.append(layers.Multiply()([linear4, s4]))
    # Prediction layer: right knee
    s5 = selection_input[:, :, 15:18]
    linear5 = get_prediction_subnet([seq_dropout, s5, time_input,
                                     delta[0], delta[2]])
    delta.append(layers.Multiply()([linear5, s5]))
    # Concatenate all predictions
    deltas = layers.Concatenate(axis=2)(delta)
    # Final prediction: pose_input + deltas
    x = layers.Add()([pose_input[:, :, :15], deltas])
    # Create the model
    model = keras.Model(inputs=[pose_input, selection_input, time_input],
                        outputs=[x])
    return model


if __name__ == "__main__":
    model = get_human_model()
    keras.utils.plot_model(model)
