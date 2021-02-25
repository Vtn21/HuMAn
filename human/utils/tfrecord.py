"""tfrecord.py

Contains tools for loading, parsing and decoding data from TFRecords files.
Such files are generated using the "preprocess_amass.py" script.

Author: Victor T. N.
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


def parse_record(record):
    """Parse a single record from the dataset.

    Args:
        record (raw dataset record): a single record from the dataset.

    Returns:
        (parsed record): a single parsed record from the dataset.
    """
    feature_names = {
        "poses": tf.io.VarLenFeature(tf.float32),
        "betas": tf.io.VarLenFeature(tf.float32),
        "dt": tf.io.FixedLenFeature([], tf.float32),
        "gender": tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(record, feature_names)


def decode_record(parsed_record):
    """Decode a previously parsed record.

    Args:
        parsed_record (raw dataset record): a single record from the dataset,
                                            previously parsed by the
                                            "parse_record" function.

    Returns:
        poses (tensor): N x 72 tensor with the sequence of poses.
        betas (tensor): 1D tensor with all shape primitives.
        dt (tensor): float tensor with the time step for this sequence.
        gender (string): gender of the subject.
    """
    poses = tf.reshape(tf.sparse.to_dense(parsed_record["poses"]), [-1, 72])
    betas = tf.reshape(tf.sparse.to_dense(parsed_record["betas"]), [1, -1])
    dt = parsed_record["dt"]
    if parsed_record["gender"] == 1:
        gender = "female"
    elif parsed_record["gender"] == -1:
        gender = "male"
    else:
        gender = "neutral"
    return poses, betas, dt, gender
