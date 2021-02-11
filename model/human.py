"""
human.py

Defines the HuMAn neural network model.

Author: Victor T. N.
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402
from tensorflow.keras import Input  # noqa: E402
from tensorflow.keras import layers  # noqa: E402


def get_human_model():
    # Input layers
    pose_input = Input(shape=(1, 72), name="pose_input")
    selection_input = Input(shape=(1, 72), name="selection_input")
    time_input = Input(shape=(1,), name="time_input")
    # Dropout on pose input (avoid overfitting)
    pose_dropout = layers.Dropout(0.1)(pose_input)
    # Concatenate all inputs
    conc_input = layers.Concatenate(axis=1)([pose_dropout,
                                             selection_input,
                                             time_input])
    # LSTM layer with dropout
    sequence_output = layers.LSTM(2048, return_sequences=True)(conc_input)
    sequence_dropout = layers.Dropout(0.2)(sequence_output)
