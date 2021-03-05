"""
load_tfrecord.py

Load the AMASS TFRecords created with "amass_to_tfrecord.py".

This script is important for testing if data loaded into the TFRecord
file retains all necessary information for training. It also demonstrates
how to join the multiple TFRecords files from each split into a single
tf.data dataset.

Author: Victor T. N.
"""


import glob
import os
from human.utils.tfrecord import decode_record, parse_record
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf  # noqa: E402


if __name__ == "__main__":
    # Path where the TFRecords are located
    tfr_home = "../../AMASS/tfrecords"
    # Data splits
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
        poses, seq_len, betas, dt, gender = decode_record(parse_record(record))
        # Show data
        print(f"poses: {poses}\n")
        print(f"seq_length: {seq_len}\n")
        print(f"betas: {betas}\n")
        print(f"dt: {dt}\n")
        print(f"gender: {gender}\n\n")
