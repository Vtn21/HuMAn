"""dataset.py

Contains tools for converting the original AMASS data in TFRecords into
datasets ready to be fed into the neural network.

TFRecords only contain poses, betas, dt and gender, and are already split
into training, validation and testing. Datasets must contain inputs and
targets.

Author: Victor T. N.
"""


import glob
import os
from human.utils.tfrecord import parse_record, decode_record
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


def load_splits(tfr_home, splits=["train", "valid", "test"]):
    """Load selected AMASS splits from TFRecords into tf.data datasets.

    Args:
        tfr_home (string): Path where TFrecord files are stored.
        splits (list, optional): AMASS splits. Must be the same names as the
                                 subdirectories of "tfr_home".
                                 Defaults to ["train", "valid", "test"].

    Returns:
        tf.data.Dataset dict: A dataset dictionary, containing a parsed dataset
                              for each split. Dictionary keys are the same as
                              split names.
    """
    raw_ds = {}
    parsed_ds = {}
    # Iterate through all splits
    for split in splits:
        # Full path to the datasets of a specific split, with wildcards
        tfr_paths = os.path.join(tfr_home, split, "*.tfrecord")
        # Expand with glob
        tfr_list = glob.glob(tfr_paths)
        # Load the TFRecords as a Dataset
        raw_ds[split] = tf.data.TFRecordDataset(
            tfr_list, num_parallel_reads=tf.data.AUTOTUNE)
        # Parse the dataset
        parsed_ds[split] = raw_ds[split].map(parse_record)
    return parsed_ds


def map_dataset(data, test=False):
    """A hybrid mapping function, used for defining "map_train" and "map_test"
    in a concise way and avoid repeating code. It is advised not to call this
    function directly, choosing between "map_train" and "map_test" instead.

    Args:
        data (tf.data): a parsed sample from the dataset.
        test (bool, optional): choose betwen testing and training.
            Defaults to False (training).

    Returns:
        tuple: (dict_inputs, targets) for training, or
            (dict_inputs, targets, dict_aux) for testing.
    """
    poses, seq_length, betas, dt, gender = decode_record(data)
    pose_input = poses[:seq_length]
    elapsed_input = dt * tf.ones(shape=[seq_length, 1], dtype=tf.float32)
    # Build a selection vector
    if test:
        # All ones
        selection_input = tf.ones(shape=tf.shape(pose_input))
    else:
        # Random
        rand_unif = tf.random.uniform(shape=[1, 72])
        offset = tf.random.uniform(shape=[], minval=-0.5, maxval=0.5)
        selection_vec = tf.math.round(
            rand_unif + offset*tf.ones(shape=[1, 72]))
        selection_input = tf.tile(selection_vec, [seq_length, 1])
    # Create a pose target with a random time horizon
    max_shift = tf.shape(poses, out_type=tf.int64)[0] - seq_length
    shift = tf.random.uniform(shape=[], minval=0, maxval=max_shift,
                              dtype=tf.int64)
    horizon_input = tf.cast(shift, tf.float32)*elapsed_input
    pose_target = tf.multiply(poses[shift:seq_length+shift], selection_input)
    inputs = {"pose_input": pose_input,
              "selection_input": selection_input,
              "elapsed_input": elapsed_input,
              "horizon_input": horizon_input}
    # Auxiliary data for testing purposes
    aux = {"betas": betas, "dt": dt, "gender": gender}
    if test:
        return inputs, pose_target, aux
    else:
        return inputs, pose_target


def map_train(data):
    """Map a parsed dataset into inputs (pose, selection, elapsed and horizon)
    and targets (pose). Call it using the dataset.map method, for training
    purposes.

    Args:
        data (tf.data): a parsed sample from the dataset.

    Returns:
        tuple: (dict_inputs, targets)
    """
    return map_dataset(data)


def map_test(data):
    """Map a parsed dataset into inputs (pose, selection, elapsed and horizon),
    targets (pose) and auxiliary values (betas, dt and gender). Call it using
    the dataset.map method, for testing purposes.

    Args:
        data (tf.data): a parsed sample from the dataset.

    Returns:
        tuple: (dict_inputs, targets, dict_aux)
    """
    return map_dataset(data, test=True)


def map_pose_input(data):
    """Map a parsed dataset into the pose input only. Used for adapting the
    Normalization layer. Call it using the dataset.map method.

    Args:
        data (tf.data): A parsed sample from the dataset.

    Returns:
        pose_input: the input poses.
    """
    poses, seq_length, _, _, _ = decode_record(data)
    pose_input = poses[:seq_length]
    return pose_input
