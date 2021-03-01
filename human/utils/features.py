"""features.py

Contains helper functions to create TensorFlow features from input values
of varied data types. These features are used to create TFRecord datasets.

Author: Victor T. N.
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


def _bytes_feature(value):
    """BytesList convertor for TFRecord.

    Args:
        value (string / byte): input value to be encoded.

    Returns:
        tf.train.Feature: input value encoded to BytesList
    """
    # Returns a bytes_list from a string / byte.
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """FloatList convertor for TFRecord.

    Args:
        value (float32 / float64): input value to be encoded.

    Returns:
        tf.train.Feature: input value encoded to FloatList
    """
    # Returns a float_list from a float / double.
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Int64List convertor for TFRecord.

    Args:
        value (bool / enum / (u)int32 / (u)int64): input value to be encoded.

    Returns:
        tf.train.Feature: input value encoded to Int64List
    """
    # Returns an int64_list from a bool / enum / int / uint.
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
