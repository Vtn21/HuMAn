"""
load_tfrecord.py

Load the AMASS TFRecords created with "amass_to_tfrecord.py".
This script is important for testing if data loaded into the TFRecord
    file retains all necessary information for training.

Author: Victor T. N.
"""

import glob
import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


def parse_record(record):
    """Parse a single record from the dataset.

    Args:
        record (raw dataset record): a single record from the dataset.

    Returns:
        (parsed record): a single parsed record from the dataset.
    """
    num_names = {
        "num_poses": tf.io.FixedLenFeature([], tf.int64),
        "num_betas": tf.io.FixedLenFeature([], tf.int64)
    }
    num = tf.io.parse_single_example(record, num_names)
    feature_names = {
        "poses": tf.io.FixedLenFeature([num["num_poses"]*69], tf.float32),
        "dt": tf.io.FixedLenFeature([], tf.float32),
        "betas": tf.io.FixedLenFeature([num["num_betas"]], tf.float32),
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
        poses (tensor): N x 69 tensor with the sequence of poses.
        betas (tensor): 1D tensor with all shape primitives.
        dt (tensor): float tensor with the time step for this sequence.
        gender (string): gender of the subject.
    """
    poses = tf.reshape(parsed_record["poses"], [-1, 69])
    betas = parsed_record["betas"]
    dt = parsed_record["dt"]
    if parsed_record["gender"] == 1:
        gender = "female"
    elif parsed_record["gender"] == -1:
        gender = "male"
    else:
        gender = "neutral"
    return poses, betas, dt, gender


if __name__ == "__main__":
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Data splits (same as the filenames)
    splits = ["train", "valid", "test"]
    for split in splits:
        # Display information
        print(f"Loading \"{split}\" split\n")
        # Full path to the datasets of a specific split, with wildcards
        tfr_paths = os.path.join(tfr_home, split, "*.tfrecord")
        # Expand with glob
        tfr_list = glob.glob(tfr_paths)
        # Load the TFRecords as a Dataset
        dataset = tf.data.TFRecordDataset(tfr_list,
                                          num_parallel_reads=os.cpu_count())
        # Shuffle the dataset
        dataset = dataset.shuffle(1000,
                                  reshuffle_each_iteration=True)
        # Take a record as example
        record = next(iter(dataset))
        # Parse and decode
        poses, betas, dt, gender = decode_record(parse_record(record))
        # Show data
        print(f"Poses: {poses}\n")
        print(f"dt: {dt}\n")
        print(f"betas: {betas}\n")
        print(f"gender: {gender}\n\n")
        time.sleep(1)
