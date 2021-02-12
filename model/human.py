"""
human.py

Defines the HuMAn neural network model.

Author: Victor T. N.
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402


def prediction_subnet(selection, other_inputs, name="joint"):
    conc = layers.Concatenate(axis=2, name=f"{name}_concat")(
        [selection] + other_inputs)
    dense1 = layers.Dense(128, activation="relu", name=f"{name}_dense1")(conc)
    dense2 = layers.Dense(64, activation="relu", name=f"{name}_dense2")(dense1)
    linear = layers.Dense(3, name=f"{name}_linear")(dense2)
    return layers.Multiply(name=f"{name}_multiply")([linear, selection])


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
    concat_inputs = layers.Concatenate(axis=2, name="inputs_concat")(
        [selection_input, pose_dropout, time_input])
    # LSTM layer with dropout
    seq_output = layers.LSTM(2048, return_sequences=True, name="LSTM")(
        concat_inputs)
    seq_dropout = layers.Dropout(0.2, name="LSTM_dropout")(seq_output)
    # Prediction layers
    delta = []
    # Prediction layer: root orientation
    s0 = selection_input[:, :, :3]
    delta.append(prediction_subnet(s0, [seq_dropout, time_input],
                                   name="root"))
    # Prediction layer: left hip
    s1 = selection_input[:, :, 3:6]
    delta.append(prediction_subnet(s1, [seq_dropout, time_input,
                                        delta[0]], name="hipL"))
    # Prediction layer: right hip
    s2 = selection_input[:, :, 6:9]
    delta.append(prediction_subnet(s2, [seq_dropout, time_input,
                                        delta[0]], name="hipR"))
    # Prediction layer: lower spine
    s3 = selection_input[:, :, 9:12]
    delta.append(prediction_subnet(s3, [seq_dropout, time_input,
                                        delta[0]], name="spine3"))
    # Prediction layer: left knee
    s4 = selection_input[:, :, 12:15]
    delta.append(prediction_subnet(s4, [seq_dropout, time_input,
                                        delta[0], delta[1]], name="kneeL"))
    # Prediction layer: right knee
    s5 = selection_input[:, :, 15:18]
    delta.append(prediction_subnet(s5, [seq_dropout, time_input,
                                        delta[0], delta[2]], name="kneeR"))
    # Prediction layer: mid spine
    s6 = selection_input[:, :, 18:21]
    delta.append(prediction_subnet(s6, [seq_dropout, time_input,
                                        delta[0], delta[3]], name="spine6"))
    # Prediction layer: left ankle
    s7 = selection_input[:, :, 21:24]
    delta.append(prediction_subnet(s7, [seq_dropout, time_input,
                                        delta[0], delta[1], delta[4]],
                                   name="ankleL"))
    # Prediction layer: right ankle
    s8 = selection_input[:, :, 24:27]
    delta.append(prediction_subnet(s8, [seq_dropout, time_input,
                                        delta[0], delta[2], delta[5]],
                                   name="ankleR"))
    # Prediction layer: high spine
    s9 = selection_input[:, :, 27:30]
    delta.append(prediction_subnet(s9, [seq_dropout, time_input,
                                        delta[0], delta[3], delta[6]],
                                   name="spine9"))
    # Prediction layer: left foot
    s10 = selection_input[:, :, 30:33]
    delta.append(prediction_subnet(s10, [seq_dropout, time_input, delta[0],
                                         delta[1], delta[4], delta[7]],
                                   name="footL"))
    # Prediction layer: right foot
    s11 = selection_input[:, :, 33:36]
    delta.append(prediction_subnet(s11, [seq_dropout, time_input, delta[0],
                                         delta[2], delta[5], delta[8]],
                                   name="footR"))
    # Concatenate all predictions
    deltas = layers.Concatenate(axis=2)(delta)
    # Final prediction: pose_input + deltas
    x = layers.Add()([pose_input[:, :, :36], deltas])
    # Create the model
    model = keras.Model(inputs=[pose_input, selection_input, time_input],
                        outputs=[x])
    return model


if __name__ == "__main__":
    model = get_human_model()
    keras.utils.plot_model(model, show_shapes=False)
