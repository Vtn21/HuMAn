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


def get_human_model():
    # Input layers
    pose_input = keras.Input(shape=(None, 72), name="pose_input")
    selection_input = keras.Input(shape=(None, 72), name="selection_input")
    time_input = keras.Input(shape=(None, 1), name="time_input")
    # Dropout on pose input (avoid overfitting)
    pose_dropout = layers.Dropout(0.1, name="pose_dropout")(pose_input)
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
    conc0 = layers.Concatenate(axis=2)([seq_dropout, s0, time_input])
    dense0 = layers.Dense(neurons, activation="relu")(conc0)
    linear0 = layers.Dense(3)(dense0)
    delta.append(layers.Multiply()([linear0, s0]))
    # Prediction layer: left hip
    s1 = selection_input[:, :, 3:6]
    conc1 = layers.Concatenate(axis=2)([seq_dropout, s1, time_input,
                                        delta[0]])
    dense1 = layers.Dense(neurons, activation="relu")(conc1)
    linear1 = layers.Dense(3)(dense1)
    delta.append(layers.Multiply()([linear1, s1]))
    # Prediction layer: right hip
    s2 = selection_input[:, :, 6:9]
    conc2 = layers.Concatenate(axis=2)([seq_dropout, s2, time_input,
                                        delta[0]])
    dense2 = layers.Dense(neurons, activation="relu")(conc2)
    linear2 = layers.Dense(3)(dense2)
    delta.append(layers.Multiply()([linear2, s2]))
    # Prediction layer: lower spine
    s3 = selection_input[:, :, 9:12]
    conc3 = layers.Concatenate(axis=2)([seq_dropout, s3, time_input,
                                        delta[0]])
    dense3 = layers.Dense(neurons, activation="relu")(conc3)
    linear3 = layers.Dense(3)(dense3)
    delta.append(layers.Multiply()([linear3, s3]))
    # Prediction layer: left knee
    s4 = selection_input[:, :, 12:15]
    conc4 = layers.Concatenate(axis=2)([seq_dropout, s4, time_input,
                                        delta[0], delta[1]])
    dense4 = layers.Dense(neurons, activation="relu")(conc4)
    linear4 = layers.Dense(3)(dense4)
    delta.append(layers.Multiply()([linear4, s4]))
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
