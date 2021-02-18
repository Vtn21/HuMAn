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
            tfr_list, num_parallel_reads=os.cpu_count())
        # Parse the dataset
        parsed_ds[split] = raw_ds[split].map(parse_record)
    return parsed_ds


# TODO: write functions to create all four datasets
def create_pose_time_datasets(parsed_ds):
    # Function to map into the dataset
    def pose_input_target_time(data):
        poses, _, dt, _ = decode_record(data)
        pose_input = poses[:-1]
        pose_target = poses[1:]
        return pose_input, pose_target, dt

    # Instantiate datasets
    pose_input_ds = {}
    pose_target_ds = {}
    time_ds = {}
    # Iterate through the splits
    for split in parsed_ds.keys():
        pose_input_ds[split], pose_target_ds[split], time_ds[split] = \
            parsed_ds[split].map(pose_input_target_time)
    return pose_input_ds, pose_target_ds, time_ds
