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


def load_all_splits(tfr_home, splits=["train", "valid", "test"]):
    """Load all AMASS splits from TFRecords into tf.data datasets.

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


def map_dataset(data):
    """Map a parsed dataset into inputs (pose, selection and time) and targets
    (pose). Call it using the dataset.map method.

    Args:
        data (tf.data): A parsed sample from the dataset.

    Returns:
        tuple: (dict_inputs, targets)
    """
    poses, _, dt, _ = decode_record(data)
    pose_input = poses[:-1]
    time_input = dt * tf.ones(shape=(tf.shape(pose_input)[0], 1),
                              dtype=tf.float32)
    # Build a random selection vector
    rand_unif = tf.random.uniform(shape=(1, 72))
    offset = tf.random.uniform(shape=(), minval=-0.5, maxval=0.5)
    selection_vec = tf.math.round(rand_unif + offset*tf.ones(shape=(1, 72)))
    selection_input = tf.tile(selection_vec, [tf.shape(pose_input)[0], 1])
    pose_target = tf.multiply(poses[1:], selection_input)
    inputs = {"pose_input": pose_input,
              "selection_input": selection_input,
              "time_input": time_input}
    return inputs, pose_target


def map_pose_input(data):
    """Map a parsed dataset into the pose input only. Used for adapting the
    Normalization layer. Call it using the dataset.map method.

    Args:
        data (tf.data): A parsed sample from the dataset.

    Returns:
        pose_input: the input poses.
    """
    poses, _, _, _ = decode_record(data)
    pose_input = poses[:-1]
    return pose_input
