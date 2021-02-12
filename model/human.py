"""
human.py

Defines the HuMAn neural network model.
Running this script plots a graphical representation to "model.png".

Author: Victor T. N.
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402
from tensorflow.keras.regularizers import L2  # noqa: E402


L2_PENALTY = 0.001


def prediction_subnet(selection, other_inputs, name="joint"):
    conc = layers.Concatenate(axis=2, name=f"{name}_concat")(
        [selection] + other_inputs)
    dense1 = layers.Dense(128, activation="relu", name=f"{name}_dense1",
                          kernel_regularizer=L2(L2_PENALTY))(conc)
    dropout1 = layers.Dropout(0.1, name=f"{name}_dropout1")(dense1)
    dense2 = layers.Dense(64, activation="relu", name=f"{name}_dense2",
                          kernel_regularizer=L2(L2_PENALTY))(dropout1)
    dropout2 = layers.Dropout(0.1, name=f"{name}_dropout2")(dense2)
    linear = layers.Dense(3, name=f"{name}_linear")(dropout2)
    return layers.Multiply(name=f"{name}_multiply")([linear, selection])


def get_human_model():
    # Input layers
    pose_input = keras.Input(shape=(None, 72), name="pose_input")
    selection_input = keras.Input(shape=(None, 72), name="selection_input")
    time_input = keras.Input(shape=(None, 1), name="time_input")
    # Discard some joints using the selection input
    pose_select = layers.Multiply(name="pose_select")([pose_input,
                                                       selection_input])
    # Normalize the pose inputs
    pose_norm = layers.BatchNormalization(axis=2)(pose_select)
    # Dropout on pose input (avoid overfitting)
    pose_dropout = layers.Dropout(0.1, name="pose_dropout")(pose_norm)
    # Concatenate all inputs
    concat_inputs = layers.Concatenate(axis=2, name="inputs_concat")(
        [selection_input, pose_dropout, time_input])
    # LSTM layer with dropout
    seq_output = layers.LSTM(2048, return_sequences=True, name="LSTM",
                             kernel_regularizer=L2(L2_PENALTY))(concat_inputs)
    seq_dropout = layers.Dropout(0.2, name="LSTM_dropout")(seq_output)
    # Prediction layers
    delta = []
    # Prediction layer: root orientation
    s0 = selection_input[:, :, 0:3]
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
    delta.append(prediction_subnet(s4, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 1]],
                                   name="kneeL"))
    # Prediction layer: right knee
    s5 = selection_input[:, :, 15:18]
    delta.append(prediction_subnet(s5, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 2]],
                                   name="kneeR"))
    # Prediction layer: mid spine
    s6 = selection_input[:, :, 18:21]
    delta.append(prediction_subnet(s6, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3]],
                                   name="spine6"))
    # Prediction layer: left ankle
    s7 = selection_input[:, :, 21:24]
    delta.append(prediction_subnet(s7, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 1, 4]],
                                   name="ankleL"))
    # Prediction layer: right ankle
    s8 = selection_input[:, :, 24:27]
    delta.append(prediction_subnet(s8, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 2, 5]],
                                   name="ankleR"))
    # Prediction layer: high spine
    s9 = selection_input[:, :, 27:30]
    delta.append(prediction_subnet(s9, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6]],
                                   name="spine9"))
    # Prediction layer: left foot
    s10 = selection_input[:, :, 30:33]
    delta.append(prediction_subnet(s10, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 1, 4, 7]],
                                   name="footL"))
    # Prediction layer: right foot
    s11 = selection_input[:, :, 33:36]
    delta.append(prediction_subnet(s11, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 2, 5, 8]],
                                   name="footR"))
    # Prediction layer: neck
    s12 = selection_input[:, :, 36:39]
    delta.append(prediction_subnet(s12, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9]],
                                   name="neck"))
    # Prediction layer: left clavicula
    s13 = selection_input[:, :, 39:42]
    delta.append(prediction_subnet(s13, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9]],
                                   name="clavL"))
    # Prediction layer: right clavicula
    s14 = selection_input[:, :, 42:45]
    delta.append(prediction_subnet(s14, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9]],
                                   name="clavR"))
    # Prediction layer: head
    s15 = selection_input[:, :, 45:48]
    delta.append(prediction_subnet(s15, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9, 12]],
                                   name="head"))
    # Prediction layer: left shoulder
    s16 = selection_input[:, :, 48:51]
    delta.append(prediction_subnet(s16, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9, 13]],
                                   name="shoulderL"))
    # Prediction layer: right shoulder
    s17 = selection_input[:, :, 51:54]
    delta.append(prediction_subnet(s17, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9, 14]],
                                   name="shoulderR"))
    # Prediction layer: left elbow
    s18 = selection_input[:, :, 54:57]
    delta.append(prediction_subnet(s18, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9, 13, 16]],
                                   name="elbowL"))
    # Prediction layer: right elbow
    s19 = selection_input[:, :, 57:60]
    delta.append(prediction_subnet(s19, [seq_dropout, time_input] +
                                   [delta[i] for i in [0, 3, 6, 9, 14, 17]],
                                   name="elbowR"))
    # Prediction layer: left wrist
    s20 = selection_input[:, :, 60:63]
    delta.append(prediction_subnet(s20, [seq_dropout, time_input] + [delta[i]
                                   for i in [0, 3, 6, 9, 13, 16, 18]],
                                   name="wristL"))
    # Prediction layer: right wrist
    s21 = selection_input[:, :, 63:66]
    delta.append(prediction_subnet(s21, [seq_dropout, time_input] + [delta[i]
                                   for i in [0, 3, 6, 9, 14, 17, 19]],
                                   name="wristR"))
    # Prediction layer: left hand
    s22 = selection_input[:, :, 66:69]
    delta.append(prediction_subnet(s22, [seq_dropout, time_input] + [delta[i]
                                   for i in [0, 3, 6, 9, 13, 16, 18, 20]],
                                   name="handL"))
    # Prediction layer: right hand
    s23 = selection_input[:, :, 69:72]
    delta.append(prediction_subnet(s23, [seq_dropout, time_input] + [delta[i]
                                   for i in [0, 3, 6, 9, 14, 17, 19, 21]],
                                   name="handR"))
    # Concatenate all predictions
    deltas = layers.Concatenate(axis=2)(delta)
    # Final prediction: pose_input + deltas
    pose_pred = layers.Add()([pose_select, deltas])
    # Create the model
    model = keras.Model(inputs=[pose_input, selection_input, time_input],
                        outputs=[pose_pred])
    return model


if __name__ == "__main__":
    model = get_human_model()
    keras.utils.plot_model(model, show_shapes=False)
