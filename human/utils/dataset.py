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


def folder_to_dataset(folder_path="."):
    """Load all TFRecord files inside a folder into a single dataset.

    Args:
        folder_path (string): path to the folder.

    Returns:
        tf.data.Dataset: a parsed dataset containing all the data from the
                         TFRecord files inside the input folder.
    """
    # List all TFRecords inside that folder using glob
    tfr_paths = os.path.join(folder_path, "*.tfrecord")
    tfr_list = glob.glob(tfr_paths)
    return tfrecords_to_dataset(tfr_list)


def tfrecords_to_dataset(tfr_list=[]):
    """Load a set of TFRecords into a tf.data dataset.

    Args:
        tfr_list (list): list of paths to TFRecords files.

    Returns:
        tf.data.Dataset: a parsed dataset containing all the data from the
                         input TFRecord files.
    """
    # Load all TFrecords as a Dataset
    raw_ds = tf.data.TFRecordDataset(
        tfr_list, num_parallel_reads=tf.data.AUTOTUNE)
    # Parse and return the Dataset
    return raw_ds.map(parse_record)


def map_dataset(data, skeleton="random", horizon_frames=-1):
    """A dataset mapping function with args. For training, the "map_train"
    function is provided for easy usage (default args). For evaluating, it is
    advised to create lambda functions in order to specify the args, and later
    on use "map" on the dataset.

    Args:
        data (tf.data): a parsed sample from the dataset.
        skeleton (string, optional): the skeleton structure type. Accepts
                                     "arms", "full_body", "legs", "legs_arms",
                                     and "random". Defaults to "random".
        horizon_frame (int, optional): the number of frames to shift in order
                                       to create the target predictions.
                                       Defaults to -1, that creates random
                                       shifts.

    Returns:
        tuple: (dict_inputs, targets).
    """
    poses, seq_length, _, dt, _ = decode_record(data)
    pose_input = poses[:seq_length]
    elapsed_input = dt * tf.ones(shape=[seq_length, 1], dtype=tf.float32)
    # Build a selection vector, according to the selected skeleton structure
    if skeleton == "arms":
        selection_vec = tf.concat([tf.zeros(shape=(1, 48)),
                                   tf.ones(shape=(1, 24))], axis=1)
        selection_input = tf.tile(selection_vec, [seq_length, 1])
    elif skeleton == "full_body":
        selection_input = tf.ones(shape=tf.shape(pose_input))
    elif skeleton == "legs":
        selection_vec = tf.concat([tf.tile(tf.constant(
            [[0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=tf.float32), [1, 4]),
            tf.zeros(shape=(1, 36))], axis=1)
        selection_input = tf.tile(selection_vec, [seq_length, 1])
    elif skeleton == "legs_arms":
        selection_vec = tf.concat([tf.tile(tf.constant(
            [[0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=tf.float32), [1, 4]),
            tf.zeros(shape=(1, 12)), tf.ones(shape=(1, 24))], axis=1)
        selection_input = tf.tile(selection_vec, [seq_length, 1])
    elif skeleton == "random":
        rand_unif = tf.random.uniform(shape=[1, 72])
        offset = tf.random.uniform(shape=[], minval=-0.5, maxval=0.5)
        selection_vec = tf.math.round(
            rand_unif + offset*tf.ones(shape=[1, 72]))
        selection_input = tf.tile(selection_vec, [seq_length, 1])
    else:
        return ValueError("Invalid skeleton configuration. Valid choices:"
                          "arms, full_body, legs, legs_arms, random.")
    # Compute the maximum allowed prediction horizon frame shift
    max_shift = tf.shape(poses, out_type=tf.int64)[0] - seq_length
    # Set the desired prediction horizon, in frames
    if horizon_frames == -1:
        # Create a pose target with a random time horizon
        shift = tf.random.uniform(shape=[], minval=1, maxval=max_shift,
                                  dtype=tf.int64)
    else:
        # Fixed prediction horizon (input-defined)
        shift = horizon_frames
    if horizon_frames > max_shift:
        # Return "horizon_input" filled with zeros as a warning
        # (not enough frames to shift)
        inputs = {"pose_input": pose_input,
                  "selection_input": selection_input,
                  "elapsed_input": elapsed_input,
                  "horizon_input": tf.zeros(shape=tf.shape(elapsed_input),
                                            dtype=tf.float32)}
        pose_target = pose_input
    else:
        pose_target = tf.multiply(poses[shift:seq_length+shift],
                                  selection_input)
        horizon_input = tf.cast(shift, tf.float32)*elapsed_input
        inputs = {"pose_input": pose_input,
                  "selection_input": selection_input,
                  "elapsed_input": elapsed_input,
                  "horizon_input": horizon_input}
    return inputs, pose_target


def map_train(data):
    """Map a parsed dataset into inputs (pose, selection, elapsed and horizon)
    and targets (pose). Call it using the dataset.map method, for training
    purposes. It calls "map_dataset" with default args, which randomizes
    skeleton structure and prediction horizon.

    Args:
        data (tf.data): a parsed sample from the dataset.

    Returns:
        tuple: (dict_inputs, targets)
    """
    return map_dataset(data)


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
