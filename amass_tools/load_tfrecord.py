"""
load_amass.py

Load the AMASS TFRecords created with "load_amass.py".
This script is important for testing if data loaded into the TFRecord
    file retains all necessary information for training.

Author: Victor T. N.
"""

import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf


def parse_nums(record):
    feature_names = {
        "num_poses": tf.io.FixedLenFeature([], tf.int64),
        "num_betas": tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(record, feature_names)


def parse_record(record, num_poses, num_betas):
    feature_names = {
        "poses": tf.io.FixedLenFeature([num_poses*69], tf.float32),
        "dt": tf.io.FixedLenFeature([], tf.float32),
        "betas": tf.io.FixedLenFeature([num_betas], tf.float32),
        "gender": tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(record, feature_names)


def decode_record(record):
    p_nums = parse_nums(record)
    p_record = parse_record(record, p_nums["num_poses"], p_nums["num_betas"])
    poses = tf.reshape(p_record["poses"], [-1, 69])
    betas = p_record["betas"]
    dt = p_record["dt"]
    if p_record["gender"] == 1:
        gender = "female"
    elif p_record["gender"] == -1:
        gender = "male"
    else:
        gender = "neutral"
    return poses, betas, dt, gender


if __name__ == "__main__":
    # Path where the TFRecords are located
    tfr_path = "../../AMASS/tfrecords"
    # Data splits (same as the filenames)
    # splits = ["train", "valid", "test"]
    splits = ["test"]
    for split in splits:
        # Full path to a single file
        tfr_file_path = os.path.join(tfr_path, split + ".tfrecord")
        # Load the TFRecord as a Dataset
        dataset = tf.data.TFRecordDataset(tfr_file_path)
        # Shuffle the dataset
        dataset = dataset.shuffle(1000,
                                  reshuffle_each_iteration=True)
        for record in dataset:
            # Parse and decode
            poses, betas, dt, gender = decode_record(record)
            print(poses)
            print(dt)
            print(betas)
            print(gender)
            print("\n\n")
            time.sleep(1)
