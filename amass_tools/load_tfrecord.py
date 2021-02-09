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


def parse_record(record):
    feature_names = {
        "poses": tf.io.FixedLenFeature([], tf.float32),
        "dt": tf.io.FixedLenFeature([], tf.float32),
        "betas": tf.io.FixedLenFeature([], tf.float32),
        "gender": tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(record, feature_names)


def decode_record(record):
    poses = tf.reshape(record["poses"], [-1, 69])
    dt = record["dt"]
    betas = record["betas"]
    if record["gender"] == 1:
        gender = "female"
    elif record["gender"] == -1:
        gender = "male"
    else:
        gender = "neutral"
    return poses, dt, betas, gender


if __name__ == "__main__":
    # Path where the TFRecords are located
    tfr_path = "../../AMASS/tfrecords"
    # Data splits (same as the filenames)
    splits = ["train", "valid", "test"]
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
            poses, dt, betas, gender = decode_record(parse_record(record))
            print(poses)
            print(dt)
            print(betas)
            print(gender)
            time.sleep(1)
            break
