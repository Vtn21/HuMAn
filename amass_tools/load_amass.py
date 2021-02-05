"""
load_amass.py

Load the downloaded AMASS database (.npz files) into TFRecords.
This is the preferred binary file for TensorFlow, and has performance
    advantages over loading the .npz files one by one.
Also, TFRecords are preferred over the tf.data.Dataset, as AMASS is large
    (23 GB of .npz files) and won't fit on RAM (at least not on my computer :D)

Author: Victor T. N.
"""


import concurrent.futures
import glob
import numpy as np
import os
from tqdm import trange
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide unnecessary TF messages
import tensorflow as tf


"""
TFRecord conversion functions

The following three functions convert standard TensorFlow types into TFRecord
types: BytesList, FloatList and Int64List.

https://www.tensorflow.org/tutorials/load_data/tfrecord
"""


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


def amass_example(bdata, framerate_drop=1, max_betas=10):
    """Create a TF Example from an AMASS recording.
    There is the option to drop down the framerate (augmentation).

    Args:
        bdata (numpy dict): uncompressed numpy data from the recording.
        framerate_drop (int, optional): Used to drop the original framerate
                                        (data augmentation). Defaults to 1.
        max_betas (int, optional): Limits the maximum number of shape
                                   components. Defaults to 10.

    Returns:
        [tf.train.Example]: relevant data fields wrapped into TF Example.
    """
    # Turning string gender into integer
    if str(bdata["gender"]) == "male":
        gender_int = -1
    elif str(bdata["gender"]) == "female":
        gender_int = 1
    else:
        gender_int = 0
    # Ensure that the number of betas is not greater than what is available
    num_betas = min(max_betas, bdata["betas"].shape[0])
    # Build feature dict
    feature = {
        # Keep only joint poses, discarding body orientation
        # framerate_drop acts here, picking just part of the array
        "poses": _float_feature(
            bdata["poses"][0::framerate_drop, 3:72].flatten()),
        # Time interval between recordings, considering framerate drop
        "dt": _float_feature([framerate_drop/bdata["mocap_framerate"]]),
        # Shape components (betas), up to the maximum defined amount
        "betas": _float_feature(bdata["betas"][:num_betas]),
        # Gender encoded into integer
        "gender": _int64_feature([gender_int])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(tfr_file_path, amass_path, sub_ds):
    # Create path with wildcards
    npz_glob = os.path.join(amass_path, sub_ds, "*/*.npz")
    # Create a list with all .npz file names
    npz_file_list = glob.glob(npz_glob)
    with tf.io.TFRecordWriter(tfr_file_path) as writer:
        # Iterate through the .npz files of this sub-dataset
        for i in trange(len(npz_file_list), desc=sub_ds):
            # Try to load specified file
            try:
                bdata = np.load(npz_file_list[i])
            except Exception as ex:
                print(ex)
                print("Error loading " + npz_file_list[i] +
                      ". Skipping...")
            else:
                if "poses" not in list(bdata.keys()):
                    continue
                else:
                    tf_example = amass_example(bdata)
                    writer.write(tf_example.SerializeToString())


if __name__ == "__main__":
    # Path to the datasets
    amass_path = "../../AMASS/datasets"
    # Split the sub-datasets into training, validation and testing
    # This names must match the subfolders of "amass_path"
    # Inexistent directories will be skipped
    amass_splits = {
        "train": ["ACCAD", "BMLhandball", "BMLmovi", "BMLrub", "CMU",
                  "DFaust_67", "EKUT", "Eyes_Japan_Dataset", "KIT",
                  "MPI_Limits", "TotalCapture"],
        "valid": ["HumanEva", "MPI_HDM05", "MPI_mosh", "SFU"],
        "test": ["SSM_synced", "Transitions_mocap"]
    }
    # Path to save the TFRecords files
    tfr_path = "../../AMASS/tfrecords"
    # Create an executor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Iterate over the splits
        for split in amass_splits.keys():
            # Filename for the current split
            tfr_filename = split + ".tfrecord"
            # Full path to the TFRecords file
            tfr_file_path = os.path.join(tfr_path, tfr_filename)
            # Display information about the current split
            print("Creating TFRecord file for " + split + " split")
            # Iterate over the sub-datasets of this split
            for sub_ds in amass_splits[split]:
                # Start the executor
                executor.submit(write_tfrecord,
                                tfr_file_path, amass_path, sub_ds)
